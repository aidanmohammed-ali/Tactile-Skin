"""Loss and yaw encoding helpers for tactile pose training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


YAW_PERIOD_RAD = math.pi / 2.0


def yaw_to_vector(yaw_mod90_rad: torch.Tensor) -> torch.Tensor:
    """Encode modulo-90 yaw as a continuous unit vector."""
    return torch.stack(
        [torch.cos(4.0 * yaw_mod90_rad), torch.sin(4.0 * yaw_mod90_rad)],
        dim=-1,
    )


def vector_to_yaw(yaw_vector: torch.Tensor) -> torch.Tensor:
    """Decode [cos(4*yaw), sin(4*yaw)] to yaw in [0, pi/2)."""
    angle = torch.atan2(yaw_vector[..., 1], yaw_vector[..., 0])
    angle = torch.remainder(angle, 2.0 * math.pi)
    return angle / 4.0


def yaw_angular_error(pred_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    """Return absolute modulo-90 yaw error in radians."""
    diff = torch.remainder(pred_yaw - target_yaw + YAW_PERIOD_RAD / 2.0, YAW_PERIOD_RAD)
    diff = diff - YAW_PERIOD_RAD / 2.0
    return torch.abs(diff)


def pose_loss(
    position_pred: torch.Tensor,
    yaw_vector_pred: torch.Tensor,
    position_target: torch.Tensor,
    yaw_vector_target: torch.Tensor,
    *,
    presence_logit: torch.Tensor | None = None,
    object_present_target: torch.Tensor | None = None,
    yaw_weight: float = 0.05,
    presence_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute presence-aware pose loss.

    Position and yaw are only meaningful when a block is present. Legacy callers
    that omit ``object_present_target`` behave as if every sample is present.
    """

    if object_present_target is None:
        present = torch.ones(position_target.shape[0], dtype=position_target.dtype, device=position_target.device)
    else:
        present = object_present_target.to(dtype=position_target.dtype, device=position_target.device).view(-1)

    present_mask = present > 0.5
    if present_mask.any():
        position_loss = F.smooth_l1_loss(position_pred[present_mask], position_target[present_mask])
        yaw_pred_unit = F.normalize(yaw_vector_pred[present_mask], dim=-1, eps=1e-6)
        yaw_loss = F.mse_loss(yaw_pred_unit, yaw_vector_target[present_mask])
    else:
        zero = position_pred.sum() * 0.0
        position_loss = zero
        yaw_loss = zero

    if presence_logit is None:
        presence_loss = position_pred.sum() * 0.0
    else:
        presence_loss = F.binary_cross_entropy_with_logits(presence_logit.view(-1), present)

    total = float(presence_weight) * presence_loss + position_loss + float(yaw_weight) * yaw_loss
    return {
        "total": total,
        "position": position_loss,
        "yaw": yaw_loss,
        "presence": presence_loss,
    }

