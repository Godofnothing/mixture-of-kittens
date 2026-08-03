"""Benchmarks used DeepEP hybrid-ep gitf725d29, Megatron Core 0.19.0+git602fad0, Transformer Engine 2.17.0+git2e559f06, PyTorch 2.13.0+cu130, NCCL 2.30.7, cuBLAS 13.6.0.2, CUDA 13.0"""

# ruff: noqa: E402

import gc
import os

os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")

from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
import torch
import torch.distributed as dist
import torch.nn.functional as F

from benchmarks.utils import benchmark_bwd, benchmark_fwd, check_benchmark_correctness, get_num_local_experts, get_tflops, init_distributed
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, generate_inputs, run_reference_bf16


NUM_LOCAL_TOKENS = int(os.environ.get("NUM_LOCAL_TOKENS", 2048))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
INTERMEDIATE_DIM = int(os.environ.get("INTERMEDIATE_DIM", 3072))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
HYBRIDEP_NUM_SMS = int(os.environ.get("HYBRIDEP_NUM_SMS", 20))
HYBRIDEP_PREPROCESS_SMS = int(os.environ.get("HYBRIDEP_PREPROCESS_SMS", 108))
HYBRIDEP_RANK_CAPACITY_FACTOR = float(os.environ.get("HYBRIDEP_RANK_CAPACITY_FACTOR", 1.25))
HYBRIDEP_FUSE_PERMUTE = os.environ.get("HYBRIDEP_FUSE_PERMUTE", "1") == "1"
ENABLE_TORCH_COMPILE = False  # enabling torch.compile slows down this baseline
MODULE_NAME = "benchmark_moe"


def build_megatron_moe(world_size, num_experts, mxfp8):
    quant_recipe = None
    if mxfp8:
        quant_recipe = RecipeConfig.from_config_dict(
            {
                "matchers": {
                    "shared_bf16": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": f"{MODULE_NAME}.shared_experts.linear_fc*",
                        "config": "shared_bf16",
                    }
                },
                "configs": {
                    "shared_bf16": {
                        "transformer_engine_config_type": "TEQuantizationParams",
                        "training_recipe": {},
                    }
                },
            }
        )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_DIM,
        num_attention_heads=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=INTERMEDIATE_DIM,
        moe_router_topk=TOPK,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_permute_fusion=not HYBRIDEP_FUSE_PERMUTE,
        moe_router_fusion=True,
        moe_router_padding_for_quantization=False,
        moe_shared_expert_intermediate_size=INTERMEDIATE_DIM,
        moe_shared_expert_overlap=True,
        moe_shared_expert_gate=False,
        use_grouped_gemm_for_shared_expert=False,
        moe_shared_expert_glu_interleave_size=None,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3" if mxfp8 else None,
        fp8_recipe="mxfp8" if mxfp8 else "delayed",
        fp8_param=mxfp8,
        fp8_wgrad=mxfp8,
        quant_recipe=quant_recipe,
        use_cpu_initialization=False,
        use_transformer_engine_op_fuser=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_flex_dispatcher_num_sms=HYBRIDEP_NUM_SMS,
        moe_expert_rank_capacity_factor=HYBRIDEP_RANK_CAPACITY_FACTOR,
        moe_permute_fusion_into_hybridep=HYBRIDEP_FUSE_PERMUTE,
        moe_hybridep_num_sms_preprocessing=HYBRIDEP_PREPROCESS_SMS,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_experts,
            moe_grouped_gemm=True,
        ).mlp
    )
    with get_fp8_context(config, is_init=True):
        layer = MoELayer(config=config, submodules=submodules, name=MODULE_NAME)
    layer = layer.cuda().to(dtype=torch.bfloat16)
    if not getattr(layer.experts, "_with_fused_impl", False):
        raise RuntimeError("Transformer Engine routed-expert op fuser is not active")
    return layer


class HybridEpMegatronBenchmark:
    def __init__(self, inputs, world_size, mxfp8):
        (
            x,
            self.topk_experts,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            self.d_output,
        ) = inputs
        self.num_local_experts = w_routed_gate.shape[0]
        self.layer = build_megatron_moe(world_size, self.num_local_experts * world_size, mxfp8)
        self.x = x.detach().view(NUM_LOCAL_TOKENS, 1, HIDDEN_DIM).requires_grad_()
        self.router_weights = router_weights.detach().requires_grad_()
        self.routed_fc1 = tuple(
            getattr(self.layer.experts.linear_fc1, f"weight{expert_idx}")
            for expert_idx in range(self.num_local_experts)
        )
        self.routed_fc2 = tuple(
            getattr(self.layer.experts.linear_fc2, f"weight{expert_idx}")
            for expert_idx in range(self.num_local_experts)
        )
        self.shared_fc1 = self.layer.shared_experts.linear_fc1.weight
        self.shared_fc2 = self.layer.shared_experts.linear_fc2.weight
        self.grad_inputs = (
            self.x,
            self.router_weights,
            *self.routed_fc1,
            *self.routed_fc2,
            self.shared_fc1,
            self.shared_fc2,
        )

        with torch.no_grad():
            self.shared_fc1.copy_(torch.cat((w_shared_gate, w_shared_up), dim=0))
            self.shared_fc2.copy_(w_shared_down)
            for expert_idx, (fc1, fc2) in enumerate(zip(self.routed_fc1, self.routed_fc2, strict=True)):
                source_fc1 = torch.cat((w_routed_gate[expert_idx], w_routed_up[expert_idx]), dim=0)
                if mxfp8:
                    fc1.quantize_(source_fc1)
                    fc2.quantize_(w_routed_down[expert_idx])
                else:
                    fc1.copy_(source_fc1)
                    fc2.copy_(w_routed_down[expert_idx])

    def run_fwd(self):
        dense_probs = self.router_weights.new_zeros(NUM_LOCAL_TOKENS, self.layer.config.num_moe_experts)
        dense_probs.scatter_(1, self.topk_experts, self.router_weights)
        routing_map = torch.zeros(
            NUM_LOCAL_TOKENS,
            self.layer.config.num_moe_experts,
            dtype=torch.bool,
            device=self.x.device,
        )
        routing_map.scatter_(1, self.topk_experts, True)

        with get_fp8_context(self.layer.config):
            shared_output = self.layer.shared_experts_compute(self.x)
            hidden_states, dispatched_probs = self.layer.preprocess(self.x, dense_probs, routing_map)
            dispatched_input, dispatched_probs = self.layer.dispatch(hidden_states, dispatched_probs)
            output, _ = self.layer.routed_experts_compute(dispatched_input, dispatched_probs)
            output = self.layer.combine(output)
            output = self.layer.postprocess(output, shared_output).squeeze(1)
        return output, output

    def run_bwd(self, output):
        shared_stream = self.layer.shared_experts.stream
        if shared_stream is not None:
            shared_stream.wait_stream(torch.cuda.current_stream())
        with get_fp8_context(self.layer.config):
            gradients = torch.autograd.grad(output, self.grad_inputs, self.d_output)
        if shared_stream is not None:
            torch.cuda.current_stream().wait_stream(shared_stream)

        d_fc1 = torch.stack(gradients[2 : 2 + self.num_local_experts])
        d_gate, d_up = d_fc1.split(INTERMEDIATE_DIM, dim=1)
        d_shared_gate, d_shared_up = gradients[-2].split(INTERMEDIATE_DIM, dim=0)
        return (
            gradients[0].squeeze(1),
            gradients[1],
            d_gate,
            d_up,
            torch.stack(gradients[2 + self.num_local_experts : 2 + 2 * self.num_local_experts]),
            d_shared_gate,
            d_shared_up,
            gradients[-1],
        )

    def assert_no_dispatch_overflow(self):
        global_over_budget = self.layer.token_dispatcher.check_over_budget().to(dtype=torch.int32)
        dist.all_reduce(global_over_budget, op=dist.ReduceOp.MAX)
        if global_over_budget.item():
            raise RuntimeError("HybridEP dropped tokens after exceeding its static rank budget")


def main():
    rank, world_size, device = init_distributed()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
    )
    set_experimental_flag(True)
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM}")
        print(f"HybridEP: comm SMs={HYBRIDEP_NUM_SMS}, preprocess SMs={HYBRIDEP_PREPROCESS_SMS}, rank capacity={HYBRIDEP_RANK_CAPACITY_FACTOR}, fused permute={HYBRIDEP_FUSE_PERMUTE}")
        print(f"torch.compile={ENABLE_TORCH_COMPILE} (max-autotune-no-cudagraphs)")

    variants = (("HybridEP + Megatron Core BF16", False, BF16_TOLERANCE), ("HybridEP + Megatron Core MXFP8", True, MXFP8_TOLERANCE))
    for name, mxfp8, tolerance in variants:
        benchmark = HybridEpMegatronBenchmark(inputs, world_size, mxfp8)
        run_fwd = benchmark.run_fwd
        run_bwd = benchmark.run_bwd
        if ENABLE_TORCH_COMPILE:
            run_fwd = torch.compile(run_fwd, mode="max-autotune-no-cudagraphs")
            run_bwd = torch.compile(run_bwd, mode="max-autotune-no-cudagraphs")
        reference = run_reference_bf16(*inputs)
        check_benchmark_correctness(name, run_fwd, run_bwd, reference, tolerance, rank)
        del reference

        fwd_ms = benchmark_fwd(run_fwd, device)
        bwd_ms = benchmark_bwd(run_fwd, run_bwd, device)
        benchmark.assert_no_dispatch_overflow()
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            bwd_tflops = get_tflops(bwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM, backward=True)
            print(f"{name}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s; backward {bwd_ms:.3f} ms, {bwd_tflops:.1f} TFLOP/s")
        del benchmark
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()
    from megatron.core.transformer.moe import fused_a2a

    fused_a2a.reset_hybrid_ep_buffer()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
