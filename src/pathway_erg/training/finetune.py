"""Stage C — supervised fine-tuning from a joint SSL checkpoint (plan §14.8).

``init_from_ssl`` copies the pretrained stems/fusion/router/aggregators
into a fresh ``PathModel`` and (optionally) freezes them for a number of
epochs so the new task heads warm up before encoder unfreezing (plan
§14.8: freeze 5–10 epochs, then unfreeze gradually).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_ssl_checkpoint(path: Path | str) -> dict:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    if "model" not in ckpt or "heads" not in ckpt:
        raise ValueError(f"{path} is not a joint SSL checkpoint (no model/heads)")
    return ckpt


def init_from_ssl(model: torch.nn.Module, path: Path | str) -> None:
    """Copy encoder weights from an SSL checkpoint into a fresh model.

    Only keys present in both state dicts are copied (SSL projection/decoder
    heads and task heads are never carried over — plan §14.4/14.8).  Missing
    core encoder keys are an error, not a silent skip.
    """
    ckpt = load_ssl_checkpoint(path)
    pretrained = ckpt["model"]
    missing = set(pretrained) - set(model.state_dict())
    core = {"raw_stem", "ot_stem", "fusion", "router", "comp_to_eye",
            "intensity_to_eye", "eye_to_unit"}
    if any(any(k.startswith(c + ".") or k == c for c in core) for k in missing):
        raise ValueError(
            f"SSL checkpoint misses core encoder keys: {sorted(missing)[:10]}"
        )
    compatible = {k: v for k, v in pretrained.items()
                  if k in model.state_dict()
                  and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)


def freeze_encoders(model: torch.nn.Module, freeze: bool = True) -> None:
    """Freeze/unfreeze the shared encoder; heads always stay trainable."""
    core = {"raw_stem", "ot_stem", "fusion", "router", "comp_to_eye",
            "intensity_to_eye", "eye_to_unit"}
    for name, param in model.named_parameters():
        if name.split(".")[0] in core:
            param.requires_grad = not freeze
