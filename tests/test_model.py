"""
test_model.py
-------------
Unit tests for ForecastBustUNet, loss functions, and confidence map.
All model tests are skipped gracefully if PyTorch is not installed.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SKIP_NO_TORCH = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@SKIP_NO_TORCH
def test_model_output_shapes():
    """Forward pass produces (B,1,H,W) tensors for both heads."""
    from model.architecture import ForecastBustUNet
    model = ForecastBustUNet(in_channels=6, encoder_channels=(32, 64, 128, 256))
    model.eval()
    x = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        err, prob = model(x)
    assert err.shape  == (2, 1, 64, 64), f"Error map shape: {err.shape}"
    assert prob.shape == (2, 1, 64, 64), f"Bust prob shape: {prob.shape}"


@SKIP_NO_TORCH
def test_error_map_nonnegative():
    """Error map uses Softplus — must be strictly non-negative."""
    from model.architecture import ForecastBustUNet
    model = ForecastBustUNet(in_channels=4, encoder_channels=(16, 32, 64, 128))
    model.eval()
    with torch.no_grad():
        err, _ = model(torch.randn(1, 4, 32, 32))
    assert (err >= 0).all(), "Error map has negative values"


@SKIP_NO_TORCH
def test_bust_prob_range():
    """Bust probability uses Sigmoid — must be in [0, 1]."""
    from model.architecture import ForecastBustUNet
    model = ForecastBustUNet(in_channels=4, encoder_channels=(16, 32, 64, 128))
    model.eval()
    with torch.no_grad():
        _, prob = model(torch.randn(1, 4, 32, 32))
    assert (prob >= 0).all() and (prob <= 1).all()


@SKIP_NO_TORCH
def test_confidence_map_range():
    """Confidence map must be clamped to [0, 100]."""
    from model.architecture import compute_confidence_map
    err  = torch.abs(torch.randn(2, 1, 16, 16)) * 10
    prob = torch.rand(2, 1, 16, 16)
    conf = compute_confidence_map(err, prob, clim_sigma=10.0)
    assert (conf >= 0).all() and (conf <= 100).all()


@SKIP_NO_TORCH
def test_focal_loss_positive_no_nan():
    """Focal loss must be non-negative and finite."""
    from model.loss import FocalLoss
    fl = FocalLoss(gamma=2.0, alpha=0.25)
    pred   = torch.tensor([[[[0.9, 0.1, 0.8, 0.2]]]])
    target = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    loss = fl(pred, target)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


@SKIP_NO_TORCH
def test_multitask_loss_keys():
    """MultiTaskBustLoss returns a scalar and a dict with expected keys."""
    from model.loss import MultiTaskBustLoss
    criterion = MultiTaskBustLoss(alpha=0.4, focal_gamma=2.0)
    B, H, W = 2, 16, 16
    pred_err  = torch.abs(torch.randn(B, 1, H, W))
    true_err  = torch.abs(torch.randn(B, 1, H, W))
    pred_prob = torch.rand(B, 1, H, W)
    true_bust = (torch.rand(B, 1, H, W) > 0.85).float()
    loss, d = criterion(pred_err, true_err, pred_prob, true_bust)
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert {"total", "regression", "classification"} <= d.keys()


@SKIP_NO_TORCH
def test_model_non_power_of_two_dims():
    """Model must handle non-power-of-2 spatial dimensions via padding."""
    from model.architecture import ForecastBustUNet
    model = ForecastBustUNet(in_channels=4, encoder_channels=(16, 32, 64, 128))
    model.eval()
    x = torch.randn(1, 4, 48, 56)   # neither H nor W is power of 2
    with torch.no_grad():
        err, prob = model(x)
    # Output spatial dims must match input
    assert err.shape[2:] == x.shape[2:], f"Shape mismatch: {err.shape} vs {x.shape}"


@SKIP_NO_TORCH
def test_model_gradient_flows():
    """Gradients must be non-zero for both output heads (basic sanity check)."""
    from model.architecture import ForecastBustUNet
    from model.loss import MultiTaskBustLoss
    model = ForecastBustUNet(in_channels=4, encoder_channels=(16, 32, 64, 128))
    criterion = MultiTaskBustLoss()
    x        = torch.randn(1, 4, 32, 32)
    err_gt   = torch.abs(torch.randn(1, 1, 32, 32))
    bust_gt  = (torch.rand(1, 1, 32, 32) > 0.9).float()

    err_pred, prob_pred = model(x)
    loss, _ = criterion(err_pred, err_gt, prob_pred, bust_gt)
    loss.backward()

    total_grad_norm = sum(
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    )
    assert total_grad_norm > 0, "No gradients flowed through the model"
