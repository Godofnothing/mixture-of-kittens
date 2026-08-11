import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from mok import functional, ops
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
BF16_FWD_COMM_SMS = int(os.environ.get("BF16_FWD_COMM_SMS", 16))
BF16_BWD_COMM_SMS = int(os.environ.get("BF16_BWD_COMM_SMS", 16))
MXFP8_FWD_COMM_SMS = int(os.environ.get("MXFP8_FWD_COMM_SMS", 16))
MXFP8_BWD_COMM_SMS = int(os.environ.get("MXFP8_BWD_COMM_SMS", 16))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 4096))
MACROBATCH_SIZE = int(os.environ.get("MACROBATCH_SIZE", 32 * MINIBATCH_SIZE))
# The scheduler pads each local expert's token segment to 256 rows. With the
# benchmark's 384 experts at EP=2, the default 0.5 multiplier can under-size
# the schedule even for balanced routing (192 padded local segments).
SCHEDULE_CAPACITY_MULTIPLIER = float(os.environ.get("SCHEDULE_CAPACITY_MULTIPLIER", 3.0))

BF16_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=BF16_FWD_COMM_SMS,
    bwd_num_comm_sms=BF16_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
    schedule_capacity_multiplier=SCHEDULE_CAPACITY_MULTIPLIER,
)
MXFP8_CONFIG = functional.MoKConfig(
    fwd_num_comm_sms=MXFP8_FWD_COMM_SMS,
    bwd_num_comm_sms=MXFP8_BWD_COMM_SMS,
    minibatch_size=MINIBATCH_SIZE,
    macrobatch_size=MACROBATCH_SIZE,
    schedule_capacity_multiplier=SCHEDULE_CAPACITY_MULTIPLIER,
)


class MoKBenchmark:
    def __init__(self, inputs):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
            self.d_output,
        ) = inputs
        self.num_local_experts = self.w_routed_gate.shape[0]
        self.workspace = functional.get_workspace(BF16_CONFIG, dist.group.WORLD, device=self.x.device, num_local_tokens=self.x.shape[0], hidden_size=self.x.shape[1], topk=self.topk_experts.shape[1])
        self.w_gate_mxfp8 = ops.mxfp8_quantize(self.w_routed_gate, True, True)
        self.w_up_mxfp8 = ops.mxfp8_quantize(self.w_routed_up, True, True)
        self.w_down_mxfp8 = ops.mxfp8_quantize(self.w_routed_down, True, True)

    def run_bf16_fwd(self):
        schedule = functional.build_schedule(self.workspace, BF16_CONFIG, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            BF16_CONFIG, self.workspace, schedule, self.x, self.router_weights,
            self.w_shared_gate, self.w_shared_up, self.w_shared_down,
            self.w_routed_gate, self.w_routed_up, self.w_routed_down,
        )
        return output, (schedule, forward_context)

    def run_bf16_bwd(self, context):
        schedule, forward_context = context
        return functional.backward(
            BF16_CONFIG, self.workspace, schedule, forward_context, self.d_output,
            self.x, self.router_weights, self.w_shared_gate, self.w_shared_up,
            self.w_shared_down, self.w_routed_gate, self.w_routed_up,
            self.w_routed_down,
        )

    def run_mxfp8_fwd(self):
        schedule = functional.build_schedule(self.workspace, MXFP8_CONFIG, self.topk_experts, num_local_experts=self.num_local_experts)
        output, forward_context = functional.forward(
            MXFP8_CONFIG, self.workspace, schedule, self.x, self.router_weights,
            self.w_shared_gate, self.w_shared_up, self.w_shared_down,
            self.w_gate_mxfp8[:2], self.w_up_mxfp8[:2], self.w_down_mxfp8[:2],
        )
        return output, (schedule, forward_context)

    def run_mxfp8_bwd(self, context):
        schedule, forward_context = context
        return functional.backward(
            MXFP8_CONFIG, self.workspace, schedule, forward_context, self.d_output,
            self.x, self.router_weights, self.w_shared_gate, self.w_shared_up,
            self.w_shared_down, self.w_gate_mxfp8, self.w_up_mxfp8,
            self.w_down_mxfp8[2:],
        )


class BF16ForwardImpl:
    """Adaptation of YDT's BF16ForwardImpl with EP dispatch/combine around it."""

    def __init__(self, inputs):
        (
            self.x,
            self.topk_experts,
            self.router_weights,
            self.w_shared_gate,
            self.w_shared_up,
            self.w_shared_down,
            self.w_routed_gate,
            self.w_routed_up,
            self.w_routed_down,
            self.d_output,
        ) = inputs
        self.num_local_experts = self.w_routed_gate.shape[0]
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # YDT stores grouped-mm weights as [expert, input, output].
        self.up_proj = self.w_routed_gate.transpose(1, 2).contiguous()
        self.gate_proj = self.w_routed_up.transpose(1, 2).contiguous()
        self.down_proj = self.w_routed_down.transpose(1, 2).contiguous()

    def _forward(self, *, requires_grad: bool):
        x = self.x.detach().requires_grad_() if requires_grad else self.x
        w_shared_gate = self.w_shared_gate.detach().requires_grad_() if requires_grad else self.w_shared_gate
        w_shared_up = self.w_shared_up.detach().requires_grad_() if requires_grad else self.w_shared_up
        w_shared_down = self.w_shared_down.detach().requires_grad_() if requires_grad else self.w_shared_down
        w_routed_gate = self.w_routed_gate.detach().requires_grad_() if requires_grad else self.w_routed_gate
        w_routed_up = self.w_routed_up.detach().requires_grad_() if requires_grad else self.w_routed_up
        w_routed_down = self.w_routed_down.detach().requires_grad_() if requires_grad else self.w_routed_down
        up_proj = w_routed_gate.transpose(1, 2).contiguous() if requires_grad else self.up_proj
        gate_proj = w_routed_up.transpose(1, 2).contiguous() if requires_grad else self.gate_proj
        down_proj = w_routed_down.transpose(1, 2).contiguous() if requires_grad else self.down_proj

        num_local_tokens, hidden_size = x.shape
        topk = self.topk_experts.shape[1]
        num_routes = num_local_tokens * topk

        topk_experts_flat = self.topk_experts.flatten()
        destination_ranks = topk_experts_flat // self.num_local_experts
        dispatch_order = torch.argsort(destination_ranks, stable=True)
        send_counts = torch.bincount(destination_ranks, minlength=self.world_size)
        all_send_counts = torch.empty(
            self.world_size, self.world_size, dtype=torch.int64, device=x.device
        )
        dist.all_gather_into_tensor(all_send_counts, send_counts)
        send_splits = send_counts.tolist()
        recv_splits = all_send_counts[:, self.rank].tolist()
        num_recv = sum(recv_splits)

        send_x = x[dispatch_order // topk]
        send_local_experts = (topk_experts_flat[dispatch_order] % self.num_local_experts).contiguous()
        send_scores = self.router_weights.flatten()[dispatch_order].contiguous()
        recv_x = x.new_empty((num_recv, hidden_size)).requires_grad_(requires_grad)
        recv_local_experts = torch.empty(num_recv, dtype=torch.int64, device=x.device)
        recv_scores = torch.empty(num_recv, dtype=self.router_weights.dtype, device=x.device).requires_grad_(requires_grad)
        dist.all_to_all_single(recv_x, send_x, recv_splits, send_splits)
        dist.all_to_all_single(recv_local_experts, send_local_experts, recv_splits, send_splits)
        dist.all_to_all_single(recv_scores, send_scores, recv_splits, send_splits)

        # This is the YDT BF16ForwardImpl: permute by expert, grouped GEMMs,
        # score-weighted SwiGLU, then unpermute.
        expert_order = torch.argsort(recv_local_experts, stable=True)
        num_tokens_per_expert = torch.bincount(
            recv_local_experts, minlength=self.num_local_experts
        )
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        permuted_x = recv_x[expert_order]
        permuted_scores = recv_scores[expert_order]
        up = torch._grouped_mm(permuted_x, up_proj, offs=offsets)
        gate = torch._grouped_mm(permuted_x, gate_proj, offs=offsets)
        hidden = (F.silu(up) * gate * permuted_scores.unsqueeze(-1)).to(torch.bfloat16)
        permuted_output = torch._grouped_mm(hidden, down_proj, offs=offsets)
        recv_output = torch.empty_like(permuted_output)
        recv_output[expert_order] = permuted_output

        returned_output = x.new_empty((num_routes, hidden_size))
        dist.all_to_all_single(returned_output, recv_output, send_splits, recv_splits)
        routed_output = torch.empty_like(returned_output)
        routed_output[dispatch_order] = returned_output

        shared_hidden = F.silu(x @ w_shared_gate.T) * (x @ w_shared_up.T)
        shared_output = shared_hidden @ w_shared_down.T
        output = (shared_output.float() + routed_output.view(num_local_tokens, topk, hidden_size).float().sum(dim=1)).to(torch.bfloat16)
        if not requires_grad:
            return output
        return output, (
            recv_x, recv_scores, permuted_output, expert_order, dispatch_order,
            send_splits, recv_splits, x, w_routed_gate, w_routed_up,
            w_routed_down, shared_output, w_shared_gate, w_shared_up,
            w_shared_down,
        )

    def forward(self) -> torch.Tensor:
        return self._forward(requires_grad=False)

    def run_fwd(self):
        return self._forward(requires_grad=True)

    def run_bwd(self, context):
        (
            recv_x, recv_scores, permuted_output, expert_order, dispatch_order,
            send_splits, recv_splits, x, w_routed_gate, w_routed_up,
            w_routed_down, shared_output, w_shared_gate, w_shared_up,
            w_shared_down,
        ) = context
        num_local_tokens, hidden_size = x.shape
        topk = self.topk_experts.shape[1]
        num_routes = num_local_tokens * topk

        d_flat_output = self.d_output.unsqueeze(1).expand(-1, topk, -1).reshape(num_routes, hidden_size)
        d_returned_output = d_flat_output[dispatch_order]
        d_recv_output = recv_x.new_empty(recv_x.shape)
        dist.all_to_all_single(d_recv_output, d_returned_output, recv_splits, send_splits)
        d_permuted_output = d_recv_output[expert_order]
        d_recv_x, d_w_routed_gate, d_w_routed_up, d_w_routed_down, d_recv_scores = torch.autograd.grad(
            permuted_output,
            (recv_x, w_routed_gate, w_routed_up, w_routed_down, recv_scores),
            d_permuted_output,
        )

        returned_d_x = x.new_empty((num_routes, hidden_size))
        dist.all_to_all_single(returned_d_x, d_recv_x, send_splits, recv_splits)
        flat_d_x = torch.empty_like(returned_d_x)
        flat_d_x[dispatch_order] = returned_d_x
        d_x_routed = flat_d_x.view(num_local_tokens, topk, hidden_size).float().sum(dim=1)

        returned_d_scores = self.router_weights.new_empty((num_routes,))
        dist.all_to_all_single(returned_d_scores, d_recv_scores, send_splits, recv_splits)
        flat_d_scores = torch.empty_like(returned_d_scores)
        flat_d_scores[dispatch_order] = returned_d_scores
        d_router_weights = flat_d_scores.view(num_local_tokens, topk)

        d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = torch.autograd.grad(
            shared_output,
            (x, w_shared_gate, w_shared_up, w_shared_down),
            self.d_output,
        )
        d_x = (d_x_routed + d_x_shared.float()).to(torch.bfloat16)
        return (
            d_x, d_router_weights, d_w_routed_gate, d_w_routed_up,
            d_w_routed_down, d_w_shared_gate, d_w_shared_up, d_w_shared_down,
        )


def main() -> None:
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    benchmark = MoKBenchmark(inputs)
    ydt_baseline = BF16ForwardImpl(inputs)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}")
        print(f"BF16 comm SMs={BF16_FWD_COMM_SMS}/{BF16_BWD_COMM_SMS}, MXFP8 comm SMs={MXFP8_FWD_COMM_SMS}/{MXFP8_BWD_COMM_SMS}, minibatch={MINIBATCH_SIZE}, macrobatch={MACROBATCH_SIZE}, schedule capacity multiplier={SCHEDULE_CAPACITY_MULTIPLIER}")

    variants = (
        ("BF16", benchmark.run_bf16_fwd, benchmark.run_bf16_bwd, BF16_TOLERANCE),
        ("MXFP8", benchmark.run_mxfp8_fwd, benchmark.run_mxfp8_bwd, MXFP8_TOLERANCE),
        ("YDT BF16ForwardImpl", ydt_baseline.run_fwd, ydt_baseline.run_bwd, BF16_TOLERANCE),
    )
    reference = run_reference_bf16(*inputs)
    for precision, run_fwd, run_bwd, tolerance in variants:
        check_benchmark_correctness(precision, run_fwd, run_bwd, reference, tolerance, rank)
    del reference

    for precision, run_fwd, run_bwd, _ in variants:
        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{precision}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
