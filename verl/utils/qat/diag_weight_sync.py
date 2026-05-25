"""
Diagnostic logging for W4A4 MoE weight sync.

Add to vLLMColocateWorkerExtension.update_weights_from_ipc() to diagnose
why weights produce garbage output. Logs:
1. Which params are received in each bucket
2. Whether weight_loader is present on target params
3. Weight statistics before/after load_weights
4. quant_config state after manual_process_weights_after_loading

Usage: Set VERL_DIAG_WEIGHT_SYNC=1 environment variable before training.
"""
import logging
import os

import torch

logger = logging.getLogger("diag_weight_sync")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[DIAG] %(message)s"))
    logger.addHandler(handler)

_ENABLED = os.environ.get("VERL_DIAG_WEIGHT_SYNC", "0") == "1"


def _find_first_moe_layer(actual_model):
    """Find the first actual FusedMoE layer (not dense layers or embeddings)."""
    for name, module in actual_model.named_modules():
        # FusedMoE layers have global_num_experts attribute and w13_ prefix params
        if hasattr(module, "global_num_experts") and hasattr(module, "w13_weight_global_scale"):
            return name, module
    # Fallback: look for any module with w13_ params (but not dense layers)
    for name, module in actual_model.named_modules():
        if hasattr(module, "w13_weight_global_scale"):
            return name, module
    return None, None


def diag_before_prepare(model):
    """Call before prepare_qat_for_load_weights."""
    if not _ENABLED:
        return

    actual_model = model.model if hasattr(model, "model") else model
    name, module = _find_first_moe_layer(actual_model)

    if module is None:
        logger.info("No MoE layer found!")
        return

    qm = getattr(module, "quant_method", None)
    logger.info(f"First MoE layer: {name}")
    logger.info(f"  quant_method type: {type(qm).__name__ if qm else 'None'}")
    logger.info(f"  _process_weights_call_count: {getattr(module, '_process_weights_call_count', 'N/A')}")
    logger.info(f"  has _hf_param_meta: {hasattr(module, '_hf_param_meta')}")
    logger.info(f"  has _weight_loaders: {hasattr(module, '_weight_loaders')}")

    # Check key params
    for pname in ["w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale",
                   "w13_input_global_scale", "w2_input_global_scale",
                   "w13_weight_global_scale", "w2_weight_global_scale"]:
        p = getattr(module, pname, None)
        if p is None:
            logger.info(f"  {pname}: NOT FOUND (deleted?)")
        else:
            is_param = isinstance(p, torch.nn.Parameter)
            has_wl = hasattr(p, "weight_loader")
            extra = ""
            if p.is_floating_point() and p.numel() > 0 and p.numel() <= 1024:
                pf = p.float()
                extra = (
                    f" mean={pf.mean():.6f} max={pf.max():.6f}"
                    f" has_inf={torch.isinf(pf).any()} has_nan={torch.isnan(pf).any()}"
                    f" first3={pf.flatten()[:3].tolist()}"
                )
            logger.info(
                f"  {pname}: shape={tuple(p.shape)} dtype={p.dtype} is_param={is_param} has_wl={has_wl}{extra}"
            )

    # Also check _hf_param_meta contents
    if hasattr(module, "_hf_param_meta"):
        logger.info(f"  _hf_param_meta keys: {list(module._hf_param_meta.keys())}")


def diag_after_prepare(model):
    """Call after prepare_qat_for_load_weights."""
    if not _ENABLED:
        return

    actual_model = model.model if hasattr(model, "model") else model
    name, module = _find_first_moe_layer(actual_model)

    if module is None:
        logger.info("After prepare: No MoE layer found!")
        return

    logger.info(f"After prepare - {name}:")
    for pname in ["w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale",
                   "w13_input_global_scale", "w2_input_global_scale",
                   "w13_weight_global_scale", "w2_weight_global_scale"]:
        p = getattr(module, pname, None)
        if p is None:
            logger.info(f"  {pname}: NOT FOUND")
        else:
            has_wl = hasattr(p, "weight_loader")
            extra = ""
            if p.is_floating_point() and p.numel() > 0 and p.numel() <= 1024:
                pf = p.float()
                extra = (
                    f" mean={pf.mean():.6f} max={pf.max():.6f}"
                    f" has_inf={torch.isinf(pf).any()} has_nan={torch.isnan(pf).any()}"
                )
            logger.info(f"  {pname}: shape={tuple(p.shape)} dtype={p.dtype} has_wl={has_wl}{extra}")


def diag_on_bucket(weights, bucket_idx):
    """Call for each received bucket of weights."""
    if not _ENABLED:
        return

    expert_keys = [n for n, _ in weights if "mlp.experts" in n]
    non_expert_keys = [n for n, _ in weights if "mlp.experts" not in n]
    input_gs_keys = [n for n, _ in weights if "input_global_scale" in n]
    weight_gs_keys = [n for n, _ in weights if "weight_global_scale" in n]

    # Collect unique suffixes
    suffixes = set()
    for n, _ in weights:
        suffix = n.rsplit(".", 1)[-1] if "." in n else n
        suffixes.add(suffix)

    total_bytes = sum(t.element_size() * t.nelement() for _, t in weights)
    logger.info(
        f"Bucket {bucket_idx}: {len(weights)} tensors ({total_bytes/(1024*1024):.1f}MB), "
        f"expert={len(expert_keys)}, non_expert={len(non_expert_keys)}, "
        f"input_global_scale={len(input_gs_keys)}, weight_global_scale={len(weight_gs_keys)}, "
        f"suffixes={suffixes}"
    )

    # Log first weight_global_scale value in first bucket
    if bucket_idx == 0:
        for name, tensor in weights:
            if "weight_global_scale" in name:
                logger.info(
                    f"  First weight_global_scale: {name} = {tensor.tolist()} "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )
                break
        for name, tensor in weights:
            if "input_global_scale" in name:
                logger.info(
                    f"  First input_global_scale: {name} = {tensor.tolist()} "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )
                break

    # Check for any NaN/inf values in scales
    for name, tensor in weights:
        if "global_scale" in name:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning(f"  NaN/inf in {name}: {tensor}")
            elif tensor.abs().max() < 1e-10:
                logger.warning(f"  Near-zero {name}: {tensor}")
            break  # Only check first one


def diag_after_all_buckets(model):
    """Call after all buckets loaded but BEFORE process_weights_after_loading."""
    if not _ENABLED:
        return

    actual_model = model.model if hasattr(model, "model") else model
    name, module = _find_first_moe_layer(actual_model)

    if module is None:
        logger.info("After all buckets: No MoE layer found!")
        return

    logger.info(f"After all buckets loaded (before process) - {name}:")
    for pname in ["w13_weight_packed", "w13_weight_scale",
                   "w13_input_global_scale", "w13_weight_global_scale",
                   "w2_input_global_scale", "w2_weight_global_scale"]:
        p = getattr(module, pname, None)
        if p is None:
            logger.info(f"  {pname}: NOT FOUND")
        else:
            if p.is_floating_point():
                pf = p.float()
                logger.info(
                    f"  {pname}: shape={tuple(p.shape)} dtype={p.dtype} "
                    f"mean={pf.mean():.6f} max={pf.max():.6f} min={pf.min():.6f} "
                    f"has_inf={torch.isinf(pf).any()} has_nan={torch.isnan(pf).any()} "
                    f"first5={pf.flatten()[:5].tolist()}"
                )
            else:
                logger.info(f"  {pname}: shape={tuple(p.shape)} dtype={p.dtype}")


def diag_after_process(model):
    """Call after manual_process_weights_after_loading."""
    if not _ENABLED:
        return

    actual_model = model.model if hasattr(model, "model") else model

    for name, module in actual_model.named_modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None and not hasattr(module, "scheme") and "KVCache" not in type(qm).__name__:
            if hasattr(qm, "moe_quant_config") and qm.moe_quant_config is not None:
                qc = qm.moe_quant_config
                logger.info(f"After process - {name}:")
                for attr in ["w1_scale", "w2_scale", "g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"]:
                    val = getattr(qc, attr, None)
                    if val is None:
                        logger.info(f"  qc.{attr}: None")
                    elif isinstance(val, torch.Tensor):
                        logger.info(
                            f"  qc.{attr}: shape={tuple(val.shape)} "
                            f"mean={val.float().mean():.6f} max={val.float().max():.6f}"
                        )
                    else:
                        logger.info(f"  qc.{attr}: {val}")

                # Verify kernel's quant_config
                kernel = getattr(qm, "kernel", None)
                if kernel is not None:
                    kqc = getattr(kernel.fused_experts, "quant_config", None)
                    if kqc is not None:
                        logger.info(f"  kernel.fused_experts.quant_config is qm.moe_quant_config: {kqc is qc}")
                    else:
                        logger.warning(f"  kernel.fused_experts.quant_config is None!")
                else:
                    logger.warning(f"  qm.kernel is None!")
                break
