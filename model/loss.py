"""
loss.py
-------
MultiTaskBustLoss: Combined loss for the dual-head ForecastBustUNet.

  L_total = alpha * L_regression + (1 - alpha) * L_classification

Where:
  L_regression  = Huber (Smooth L1) loss on predicted error maps
  L_classification = Focal Loss on bust probability (handles class imbalance)

Focal Loss (Lin et al. 2017):
  FL(p_t) = -(1 - p_t)^gamma * log(p_t)
  Upweights hard-to-classify (bust) samples; gamma=2.0 is standard.

Bust events are typically 5-15% of all grid cells, so focal loss is
essential to prevent the model from predicting "never bust" everywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for binary classification.

    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Positive class weight (default 0.25). None = no class weighting.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, 1, H, W] in [0, 1]
            target: Binary ground truth [B, 1, H, W] in {0, 1}
            mask: Optional binary mask [B, 1, H, W] (1 = valid, 0 = ignore)

        Returns:
            Scalar focal loss
        """
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        # Compute p_t
        p_t = pred * target + (1.0 - pred) * (1.0 - target)

        # Focal weighting
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce

        # Apply mask
        if mask is not None:
            loss = loss * mask
            valid = mask.sum().clamp(min=1.0)
            return loss.sum() / valid

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskBustLoss(nn.Module):
    """
    Combined multi-task loss for ForecastBustUNet.

    L = alpha * Huber(error_head) + (1-alpha) * FocalLoss(bust_head)

    Args:
        alpha: Weight for regression loss (0 → classification only, 1 → regression only)
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss class weight for positive (bust) class
    """

    def __init__(
        self,
        alpha: float = 0.4,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(
        self,
        pred_error: torch.Tensor,
        true_error: torch.Tensor,
        pred_prob: torch.Tensor,
        true_bust: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_error: Predicted error magnitude [B, 1, H, W]
            true_error: Ground truth absolute error [B, 1, H, W]
            pred_prob: Predicted bust probability [B, 1, H, W]
            true_bust: Binary bust labels [B, 1, H, W]
            mask: Optional land/sea mask [B, 1, H, W]

        Returns:
            total_loss: Scalar combined loss
            loss_dict: Dict with individual loss components for logging
        """
        # Regression: Huber loss on absolute error map
        if mask is not None:
            reg_loss = F.smooth_l1_loss(
                pred_error * mask,
                true_error * mask,
                reduction="sum",
            ) / mask.sum().clamp(min=1.0)
        else:
            reg_loss = F.smooth_l1_loss(pred_error, true_error)

        # Classification: Focal loss on bust probability
        cls_loss = self.focal(pred_prob, true_bust, mask=mask)

        total_loss = self.alpha * reg_loss + (1.0 - self.alpha) * cls_loss

        loss_dict = {
            "total": total_loss.item(),
            "regression": reg_loss.item(),
            "classification": cls_loss.item(),
        }
        return total_loss, loss_dict
