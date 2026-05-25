# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mixed NVFP4 MoE Patch: w13 (gate+up) W4A4 via CUTLASS, w2 (down) W4A16 via Marlin.

Monkey-patches vLLM's CompressedTensorsMoEMethod.get_moe_method to detect
mixed quantization schemes and route to CompressedTensorsW4A4MixedNvfp4MoEMethod.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_original_get_moe_method = None
_patch_applied = False


# ---------------------------------------------------------------------------
# CompressedTensorsW4A4MixedNvfp4MoEMethod
# ---------------------------------------------------------------------------
class CompressedTensorsW4A4MixedNvfp4MoEMethod:
    """Mixed NVFP4 MoE: w13 (gate+up) W4A4 via CUTLASS, w2 (down) W4A16 via Marlin.

    Stage 1 (gate+up): FP4 activation quantization → CUTLASS FP4 GEMM
    Stage 2 (down):    BF16 activation (no quant) → Marlin W4A16 GEMM
    """

    def __init__(self, moe, layer_name=None):
        from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

        FusedMoEMethodBase.__init__(self, moe)
        self.group_size = 16

    @property
    def topk_indices_dtype(self):
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.layers.fused_moe import (
            FusedMoeWeightScaleSupported,
        )
        from vllm.model_executor.utils import set_weight_attrs

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # w13 weight (FP4 packed)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales — w13 needs it for W4A4; w2 for checkpoint compat
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.empty(num_experts, dtype=torch.float32), requires_grad=False)
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        from vllm import _custom_ops as ops
        from vllm._core_ext import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            nvfp4_marlin_process_global_scale,
            nvfp4_marlin_process_scales,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            swizzle_blockscale,
        )
        from vllm.model_executor.utils import replace_parameter

        layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data, requires_grad=False)
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data, requires_grad=False)
        delattr(layer, "w2_weight_packed")

        E = layer.num_experts
        K = layer.hidden_size
        N = layer.intermediate_size_per_partition
        device = layer.w13_weight.device
        param_dtype = layer.params_dtype

        # ===== Process w13 for CUTLASS (W4A4) =====
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0],
            layer.w13_weight_global_scale[:, 1],
        ):
            logger.warning("w1_weight_global_scale != w3_weight_global_scale, accuracy may be affected")
        w13_wgs = layer.w13_weight_global_scale[:, 0].contiguous()
        w13_scale_2 = 1.0 / w13_wgs

        a13_raw = 1.0 / layer.w13_input_global_scale
        a13_scale = a13_raw.max(dim=1).values.to(torch.float32)

        w13_scale = swizzle_blockscale(layer.w13_weight_scale)
        pad_size = w13_scale.size(1) - layer.w13_weight.size(1)
        if pad_size > 0:
            if self.moe.is_act_and_mul:
                raise NotImplementedError(
                    "Intermediate size padding with is_act_and_mul is not supported for mixed NVFP4 MoE"
                )
            w13_padded = torch.nn.functional.pad(layer.w13_weight, (0, 0, 0, pad_size))
            replace_parameter(layer, "w13_weight", w13_padded)

        replace_parameter(layer, "w13_weight_scale", w13_scale)
        layer.a1_gscale = 1.0 / a13_scale
        layer.g1_alphas = a13_scale * w13_scale_2

        # ===== Process w2 for Marlin (W4A16) =====
        w2_gs = (1.0 / layer.w2_weight_global_scale).to(param_dtype)

        perm = torch.empty(0, dtype=torch.int, device=device)
        w2_repacked_list = []
        for i in range(E):
            qweight = layer.w2_weight[i].view(torch.int32).T.contiguous()
            marlin_qw = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=N,
                size_n=K,
                num_bits=4,
                is_a_8bit=False,
            )
            w2_repacked_list.append(marlin_qw)
        w2_marlin = torch.cat([x.unsqueeze(0) for x in w2_repacked_list], 0)

        w2_scales = layer.w2_weight_scale.to(param_dtype)
        w2_scale_list = []
        for i in range(E):
            scale = w2_scales[i].T
            ms = marlin_permute_scales(
                s=scale,
                size_k=N,
                size_n=K,
                group_size=self.group_size,
                is_a_8bit=False,
            )
            ms = nvfp4_marlin_process_scales(ms)
            w2_scale_list.append(ms)
        w2_marlin_scale = torch.cat([x.unsqueeze(0) for x in w2_scale_list], 0)
        w2_marlin_gs = nvfp4_marlin_process_global_scale(w2_gs)

        replace_parameter(layer, "w2_weight", w2_marlin)
        replace_parameter(layer, "w2_weight_scale", w2_marlin_scale)
        layer.w2_weight_scale_2 = w2_marlin_gs
        layer.marlin_workspace = marlin_make_workspace_new(device, 4)
        self._marlin_quant_type = scalar_types.float4_e2m1f

    def get_fused_moe_quant_config(self, layer):
        return None

    @property
    def is_monolithic(self):
        return False

    def apply(self, layer, x, topk_weights, topk_ids):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe import moe_align_block_size

        assert layer.activation == "silu", "Only SiLU activation is supported"
        assert layer.expert_map is None, "Mixed W4A4/W4A16 MoE does not support expert parallelism"

        M, K = x.shape
        E = layer.w13_weight.shape[0]
        N = layer.intermediate_size_per_partition
        topk = topk_ids.shape[1]
        device = x.device
        out_dtype = x.dtype
        apply_rw = layer.apply_router_weight_on_input

        # ===== CUTLASS routing =====
        expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
        blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty(E, 3, dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty(E, 3, dtype=torch.int32, device=device)
        a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
        c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)

        ops.get_cutlass_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            E,
            N,
            K,
            blockscale_offsets,
        )

        # ===== Stage 1: CUTLASS W4A4 (gate + up) =====
        a_shuffled = ops.shuffle_rows(x, a_map)
        if apply_rw:
            assert topk == 1
            a_shuffled.mul_(topk_weights.to(out_dtype))

        rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
            a_shuffled,
            layer.a1_gscale,
            expert_offsets,
            blockscale_offsets,
            topk,
        )

        c1 = torch.empty(M * topk, 2 * N, device=device, dtype=out_dtype)
        ops.cutlass_fp4_moe_mm(
            c1,
            rep_a_fp4,
            layer.w13_weight,
            rep_a_blockscale,
            layer.w13_weight_scale,
            layer.g1_alphas,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        del rep_a_fp4, rep_a_blockscale

        # ===== SiLU + Mul (no FP4 quantization of intermediate) =====
        c2 = torch.empty(M * topk, N, device=device, dtype=out_dtype)
        torch.ops._C.silu_and_mul(c2, c1)
        del c1

        # ===== Unshuffle intermediate to natural token order =====
        c2 = ops.shuffle_rows(c2, c_map)

        # ===== Stage 2: Marlin W4A16 (down) =====
        block_size_m = 64
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            layer.global_num_experts,
            None,
            ignore_invalid_experts=True,
        )

        output = torch.empty(M * topk, K, device=device, dtype=out_dtype)
        output = ops.moe_wna16_marlin_gemm(
            c2,
            output,
            layer.w2_weight,
            None,
            layer.w2_weight_scale,
            None,
            layer.w2_weight_scale_2,
            None,
            None,
            None,
            layer.marlin_workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=not apply_rw,
            b_q_type=self._marlin_quant_type,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        del c2

        return output.view(M, topk, K).sum(dim=1)


# ---------------------------------------------------------------------------
# Patched get_moe_method — intercept mixed NVFP4 before hitting ValueError
# ---------------------------------------------------------------------------
def _patched_get_moe_method(quant_config, layer, layer_name):
    """Wraps the original get_moe_method to support mixed NVFP4 W4A4/W4A16."""
    try:
        return _original_get_moe_method(quant_config, layer, layer_name)
    except ValueError as e:
        if "same quantization scheme" not in str(e):
            raise

    # Original detected scheme mismatch — check for mixed NVFP4
    unfused_names = [layer_name + proj_name for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]]
    dicts = [quant_config.get_scheme_dict(layer, n) for n in unfused_names]
    gate_dict, up_dict, down_dict = dicts

    if gate_dict == up_dict and gate_dict is not None and down_dict is not None:
        gu_w = gate_dict.get("weights")
        gu_i = gate_dict.get("input_activations")
        d_w = down_dict.get("weights")
        d_i = down_dict.get("input_activations")
        if (
            quant_config._is_nvfp4_format(gu_w)
            and quant_config._is_nvfp4_format(gu_i)
            and quant_config._is_nvfp4_format(d_w)
            and d_i is None
        ):
            logger.info("Mixed NVFP4 MoE detected: gate/up=W4A4 (CUTLASS), down=W4A16 (Marlin)")
            return CompressedTensorsW4A4MixedNvfp4MoEMethod(layer.moe_config, layer_name)

    raise ValueError("All MoE projections need to have same quantization scheme but found multiple")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def apply_mixed_moe_patch():
    """Monkey-patch get_moe_method to support mixed NVFP4 W4A4/W4A16 MoE."""
    global _original_get_moe_method, _patch_applied
    if _patch_applied:
        return

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    _original_get_moe_method = CompressedTensorsMoEMethod.get_moe_method
    CompressedTensorsMoEMethod.get_moe_method = staticmethod(_patched_get_moe_method)

    _patch_applied = True
    logger.info("Applied mixed NVFP4 MoE patch (W4A4 gate/up + W4A16 down)")


__all__ = [
    "CompressedTensorsW4A4MixedNvfp4MoEMethod",
    "apply_mixed_moe_patch",
]
