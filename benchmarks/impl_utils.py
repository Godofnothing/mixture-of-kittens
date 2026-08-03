"""TorchAO-based utils used only for bench_nccl_torch and bench_deepep_torch
TorchAO 0.18.0+git55c8e6e3"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_cuda_3d,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda


@dataclass(frozen=True)
class Mxfp8Weight:
    forward_data: torch.Tensor
    forward_scales: torch.Tensor
    dgrad_data: torch.Tensor
    dgrad_scales: torch.Tensor


@torch.no_grad()
def prequantize_mxfp8_weight(weight):
    forward_data, forward_scales = mxfp8_quantize_cuda_3d(weight.transpose(-2, -1), scaling_mode="rceil", blocked_scale_output=True)
    dgrad_data, dgrad_scales = mxfp8_quantize_cuda_3d(weight, scaling_mode="rceil", blocked_scale_output=True)
    return Mxfp8Weight(forward_data, forward_scales, dgrad_data, dgrad_scales)


def quantize_mxfp8_rows(x, offsets):
    return mxfp8_quantize_2d_1x32_cutedsl(x.contiguous(), scaling_mode="rceil", offs=offsets)


def scaled_mxfp8_grouped_mm(a, b, a_scales, b_scales, offsets):
    return F.scaled_grouped_mm(
        a,
        b,
        a_scales,
        F.ScalingType.BlockWise1x32,
        b_scales,
        F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        offs=offsets,
        output_dtype=torch.bfloat16,
    )


class PrequantizedMxfp8GroupedMm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_t, forward_data, forward_scales, dgrad_data, dgrad_scales, offsets):
        x_data, x_scales = quantize_mxfp8_rows(x, offsets)
        output = scaled_mxfp8_grouped_mm(x_data, forward_data, x_scales, forward_scales, offsets)
        ctx.save_for_backward(x, dgrad_data, dgrad_scales, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, dgrad_data, dgrad_scales, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_data, grad_output_scales = quantize_mxfp8_rows(grad_output, offsets)
        grad_x = scaled_mxfp8_grouped_mm(grad_output_data, dgrad_data, grad_output_scales, dgrad_scales, offsets)

        _, grad_output_t_data, _, grad_output_t_scales = mxfp8_quantize_cuda(grad_output, rowwise=False, colwise=True, scaling_mode="rceil")
        _, x_data, _, x_scales = mxfp8_quantize_cuda(x.contiguous(), rowwise=False, colwise=True, scaling_mode="rceil")
        scale_offsets = offsets // 32
        num_groups = scale_offsets.numel()
        padded_num_groups = 2 ** (num_groups - 1).bit_length()
        if num_groups < padded_num_groups:
            scale_offsets = torch.cat((scale_offsets, scale_offsets[-1:].expand(padded_num_groups - num_groups)))
        grad_output_t_scales = triton_mx_block_rearrange_2d_K_groups(grad_output_t_scales, scale_offsets)
        x_scales = triton_mx_block_rearrange_2d_K_groups(x_scales, scale_offsets)
        grad_weight = scaled_mxfp8_grouped_mm(grad_output_t_data.transpose(0, 1), x_data, grad_output_t_scales, x_scales, offsets)
        return grad_x, grad_weight.transpose(-2, -1), None, None, None, None, None


def grouped_mm(x, weight_t, weight_mxfp8, offsets):
    if weight_mxfp8 is None:
        return F.grouped_mm(x, weight_t, offs=offsets)
    return PrequantizedMxfp8GroupedMm.apply(
        x,
        weight_t,
        weight_mxfp8.forward_data,
        weight_mxfp8.forward_scales,
        weight_mxfp8.dgrad_data,
        weight_mxfp8.dgrad_scales,
        offsets,
    )
