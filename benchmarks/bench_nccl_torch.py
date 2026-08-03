"""Benchmarks used PyTorch 2.13.0+cu130, TorchAO 0.18.0+git55c8e6e3, CUTLASS DSL 4.5.3, NCCL 2.30.7, cuBLAS 13.6.0.2, CUDA 13.0"""

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.impl_utils import grouped_mm, prequantize_mxfp8_weight
from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
MXFP8_ALIGNMENT = int(os.environ.get("MXFP8_ALIGNMENT", 128))
ENABLE_TORCH_COMPILE = True


@dataclass(frozen=True)
class RoutingSchedule:
    sort_indices: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    num_recv: int
    num_grouped: int
    recv_to_grouped: torch.Tensor
    padding_indices: torch.Tensor
    group_offsets: torch.Tensor


class NcclRouting:
    def __init__(self, x, topk_experts, num_local_experts):
        self.world_size = dist.get_world_size()
        self.num_tokens = x.shape[0]
        self.topk = topk_experts.shape[1]
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts * self.world_size
        self.num_routes = self.num_tokens * self.topk
        self.flat_experts = topk_experts.flatten()
        self.local_counts = torch.empty(self.num_experts, dtype=torch.int32, device=x.device)
        self.recv_counts_by_expert = torch.empty((self.world_size, num_local_experts), dtype=torch.int32, device=x.device)
        self.count_ones = torch.ones(self.num_routes, dtype=torch.int32, device=x.device)
        self.block_ids = torch.arange(self.num_experts, dtype=torch.int32, device=x.device)

    def build_schedule(self, expert_alignment):
        self.local_counts.zero_()
        self.local_counts.scatter_add_(0, self.flat_experts, self.count_ones)
        dist.all_to_all_single(self.recv_counts_by_expert.view(-1), self.local_counts)

        counts = self.recv_counts_by_expert
        send_counts = self.local_counts.view(self.world_size, self.num_local_experts).sum(1, dtype=torch.int32)
        recv_counts = counts.sum(1, dtype=torch.int32)
        raw_tokens_per_expert = counts.sum(0, dtype=torch.int32)
        tokens_per_expert = (raw_tokens_per_expert + expert_alignment - 1) // expert_alignment * expert_alignment
        sizes = torch.cat((send_counts, recv_counts, tokens_per_expert.sum(dtype=torch.int32).view(1))).tolist()
        send_splits = sizes[: self.world_size]
        recv_splits = sizes[self.world_size : 2 * self.world_size]
        num_recv = sum(recv_splits)
        num_grouped = sizes[-1]

        sort_indices = torch.argsort(self.flat_experts)
        counts_flat = counts.flatten()
        recv_blocks = torch.repeat_interleave(self.block_ids, counts_flat, output_size=num_recv)
        block_starts = torch.cumsum(counts_flat, 0, dtype=torch.int32) - counts_flat
        within_blocks = torch.arange(num_recv, dtype=torch.int32, device=self.flat_experts.device) - block_starts[recv_blocks]
        group_offsets = torch.cumsum(tokens_per_expert, 0, dtype=torch.int32)
        expert_starts = group_offsets - tokens_per_expert
        source_starts = torch.cumsum(counts, 0) - counts
        recv_to_grouped = (expert_starts.unsqueeze(0) + source_starts).flatten()[recv_blocks] + within_blocks
        padding_indices = torch.empty(0, dtype=torch.int64, device=self.flat_experts.device)
        if expert_alignment > 1:
            padding_mask = torch.ones(num_grouped, dtype=torch.bool, device=self.flat_experts.device)
            padding_mask[recv_to_grouped] = False
            padding_indices = torch.where(padding_mask)[0]

        return RoutingSchedule(
            sort_indices,
            send_splits,
            recv_splits,
            num_recv,
            num_grouped,
            recv_to_grouped,
            padding_indices,
            group_offsets,
        )


@dataclass(frozen=True)
class ForwardContext:
    schedule: RoutingSchedule
    grouped_x: torch.Tensor
    expert_output: torch.Tensor
    flat_output: torch.Tensor
    shared_output: torch.Tensor


class NcclTorchBenchmark:
    def __init__(self, inputs, mxfp8):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            self.d_output,
        ) = inputs
        self.mxfp8 = mxfp8
        self.expert_alignment = MXFP8_ALIGNMENT if mxfp8 else 1
        self.num_local_experts = w_routed_gate.shape[0]
        self.routing = NcclRouting(self.x, self.topk_experts, self.num_local_experts)
        self.num_tokens, self.hidden = self.x.shape
        self.intermediate = w_routed_gate.shape[1]
        self.topk = self.routing.topk
        self.num_routes = self.routing.num_routes
        self.recv_capacity = 2 * self.num_routes
        self.grouped_capacity = self.recv_capacity + self.num_local_experts * (self.expert_alignment - 1)

        self.send_x = torch.empty(self.num_routes, self.hidden, dtype=torch.bfloat16, device=self.x.device)
        self.returned_output = torch.empty_like(self.send_x)
        self.flat_output = torch.empty_like(self.send_x)
        self.d_flat_output = torch.empty(self.num_tokens, self.topk, self.hidden, dtype=torch.bfloat16, device=self.x.device)
        self.d_send_output = torch.empty_like(self.send_x)
        self.d_returned_x = torch.empty_like(self.send_x)
        self.d_flat_x = torch.empty_like(self.send_x)
        self.recv_x = torch.empty(self.recv_capacity, self.hidden, dtype=torch.bfloat16, device=self.x.device)
        self.grouped_x = torch.empty(self.grouped_capacity, self.hidden, dtype=torch.bfloat16, device=self.x.device)
        self.recv_output = torch.empty_like(self.recv_x)
        self.d_recv_output = torch.empty_like(self.recv_x)
        self.d_grouped_output = torch.empty_like(self.grouped_x)
        self.d_recv_x = torch.empty_like(self.recv_x)

        self.x_shared = self.x.detach().requires_grad_()
        self.w_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1).detach().requires_grad_()
        self.w_down = w_routed_down.detach().requires_grad_()
        self.w_fc1_mxfp8 = prequantize_mxfp8_weight(self.w_fc1) if mxfp8 else None
        self.w_down_mxfp8 = prequantize_mxfp8_weight(self.w_down) if mxfp8 else None
        self.w_fc1_t = self.w_fc1.transpose(1, 2)
        self.w_down_t = self.w_down.transpose(1, 2)
        self.w_shared_fc1 = torch.cat((w_shared_gate, w_shared_up), dim=0).detach().requires_grad_()
        self.w_shared_down = w_shared_down.detach().requires_grad_()

    def run_fwd(self):
        schedule = self.routing.build_schedule(self.expert_alignment)
        recv_x = self.recv_x[: schedule.num_recv]
        grouped_x_buffer = self.grouped_x[: schedule.num_grouped]
        recv_output = self.recv_output[: schedule.num_recv]
        token_indices = schedule.sort_indices // self.topk
        torch.index_select(self.x, 0, token_indices, out=self.send_x)
        dispatch_work = dist.all_to_all_single(recv_x, self.send_x, schedule.recv_splits, schedule.send_splits, async_op=True)

        fc1_shared = self.x_shared @ self.w_shared_fc1.T
        gate_shared, up_shared = fc1_shared.split(self.intermediate, dim=1)
        hidden_shared = F.silu(gate_shared).mul_(up_shared)
        shared_output = hidden_shared @ self.w_shared_down.T

        dispatch_work.block_current_stream()
        if self.mxfp8:
            grouped_x_buffer.index_fill_(0, schedule.padding_indices, 0)
        grouped_x_buffer.index_copy_(0, schedule.recv_to_grouped, recv_x)
        grouped_x = grouped_x_buffer.detach().requires_grad_()
        fc1 = grouped_mm(grouped_x, self.w_fc1_t, self.w_fc1_mxfp8, schedule.group_offsets)
        gate, up = fc1.split(self.intermediate, dim=1)
        hidden = F.silu(gate).mul_(up)
        expert_output = grouped_mm(hidden, self.w_down_t, self.w_down_mxfp8, schedule.group_offsets)
        torch.index_select(expert_output.detach(), 0, schedule.recv_to_grouped, out=recv_output)

        combine_work = dist.all_to_all_single(self.returned_output, recv_output, schedule.send_splits, schedule.recv_splits, async_op=True)
        combine_work.block_current_stream()
        self.flat_output.index_copy_(0, schedule.sort_indices, self.returned_output)
        flat_output = self.flat_output.view(self.num_tokens, self.topk, self.hidden).float()
        routed_output = torch.bmm(self.router_weights.unsqueeze(1), flat_output).squeeze(1)
        output = (routed_output + shared_output.float()).to(torch.bfloat16)
        return output, ForwardContext(schedule, grouped_x, expert_output, self.flat_output, shared_output)

    def run_bwd(self, context):
        schedule = context.schedule
        d_recv_output = self.d_recv_output[: schedule.num_recv]
        d_grouped_output = self.d_grouped_output[: schedule.num_grouped]
        d_recv_x = self.d_recv_x[: schedule.num_recv]
        torch.mul(self.d_output.unsqueeze(1), self.router_weights.unsqueeze(2), out=self.d_flat_output)
        torch.index_select(self.d_flat_output.view(self.num_routes, self.hidden), 0, schedule.sort_indices, out=self.d_send_output)
        dispatch_work = dist.all_to_all_single(d_recv_output, self.d_send_output, schedule.recv_splits, schedule.send_splits, async_op=True)

        d_x_shared, d_w_shared_fc1, d_w_shared_down = torch.autograd.grad(
            context.shared_output,
            (self.x_shared, self.w_shared_fc1, self.w_shared_down),
            self.d_output,
        )

        dispatch_work.block_current_stream()
        if self.mxfp8:
            d_grouped_output.index_fill_(0, schedule.padding_indices, 0)
        d_grouped_output.index_copy_(0, schedule.recv_to_grouped, d_recv_output)
        d_grouped_x, d_w_fc1, d_w_down = torch.autograd.grad(
            context.expert_output,
            (context.grouped_x, self.w_fc1, self.w_down),
            d_grouped_output,
        )
        torch.index_select(d_grouped_x, 0, schedule.recv_to_grouped, out=d_recv_x)
        combine_work = dist.all_to_all_single(self.d_returned_x, d_recv_x, schedule.send_splits, schedule.recv_splits, async_op=True)

        flat_output_t = context.flat_output.view(self.num_tokens, self.topk, self.hidden).float().transpose(1, 2)
        d_router_weights = torch.bmm(self.d_output.float().unsqueeze(1), flat_output_t).squeeze(1)

        combine_work.block_current_stream()
        self.d_flat_x.index_copy_(0, schedule.sort_indices, self.d_returned_x)
        d_x_routed = self.d_flat_x.view(self.num_tokens, self.topk, self.hidden).sum(1, dtype=torch.float32)
        d_x = (d_x_routed + d_x_shared.float()).to(torch.bfloat16)
        d_w_routed_gate, d_w_routed_up = d_w_fc1.split(self.intermediate, dim=1)
        d_w_shared_gate, d_w_shared_up = d_w_shared_fc1.split(self.intermediate, dim=0)
        return (
            d_x,
            d_router_weights,
            d_w_routed_gate,
            d_w_routed_up,
            d_w_down,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )


def main():
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}")
        print(f"torch.compile={ENABLE_TORCH_COMPILE} (max-autotune)")

    variants = (("NCCL + PyTorch BF16", False, BF16_TOLERANCE), ("NCCL + TorchAO MXFP8", True, MXFP8_TOLERANCE))
    for name, mxfp8, tolerance in variants:
        if mxfp8 and num_local_experts > 32:
            if rank == 0:
                print(f"{name} unsupported with {num_local_experts} local experts")
            continue

        benchmark = NcclTorchBenchmark(inputs, mxfp8)
        run_fwd = benchmark.run_fwd
        run_bwd = benchmark.run_bwd
        if ENABLE_TORCH_COMPILE:
            run_fwd = torch.compile(run_fwd, mode="max-autotune")
            run_bwd = torch.compile(run_bwd, mode="max-autotune")
        reference = run_reference_bf16(*inputs)
        check_benchmark_correctness(name, run_fwd, run_bwd, reference, tolerance, rank)
        del reference

        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{name}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")
        del benchmark

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
