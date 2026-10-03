"""Reference compound loss; weights remain ablation parameters."""

from __future__ import annotations


def create_paper_loss(bce_weight: float = 1.0, dice_weight: float = 1.0,
                      distance_weight: float = 0.10,
                      boundary_weight: float = 0.20,
                      boundary_band: float = 0.25):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError("paper loss requires torch") from error

    class PaperLoss(nn.Module):
        def forward(self, outputs, target_mask, target_distance):
            logits = outputs["mask_logits"]
            target_mask_float = target_mask.to(dtype=logits.dtype)
            base_bce = F.binary_cross_entropy_with_logits(logits, target_mask_float)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * target_mask_float).sum(dim=(1, 2, 3))
            denominator = probabilities.sum(dim=(1, 2, 3)) + target_mask_float.sum(dim=(1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
            distance = F.smooth_l1_loss(outputs["distance"], target_distance)
            band = target_distance.abs() <= boundary_band
            if band.any():
                boundary_bce = F.binary_cross_entropy_with_logits(
                    logits[band], target_mask_float[band])
            else:
                boundary_bce = logits.sum() * 0.0
            total = (bce_weight * base_bce + dice_weight * dice +
                     distance_weight * distance + boundary_weight * boundary_bce)
            if "auxiliary_logits" in outputs:
                auxiliary_target = F.interpolate(target_mask_float,
                    size=outputs["auxiliary_logits"].shape[-2:], mode="nearest")
                total = total + 0.10 * F.binary_cross_entropy_with_logits(
                    outputs["auxiliary_logits"], auxiliary_target)
            return {
                "loss": total,
                "bce": base_bce.detach(),
                "dice": dice.detach(),
                "distance": distance.detach(),
                "boundary_bce": boundary_bce.detach(),
            }

    return PaperLoss()

