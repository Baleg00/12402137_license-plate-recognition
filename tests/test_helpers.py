from pathlib import Path

import pytest
import torch

import model


# =========
# Utilities
# =========

def _has(name: str) -> bool:
    return hasattr(model, name)


def _tmp_checkpoint_path(tmp_path: Path, name: str = "ckpt.pth") -> Path:
    p = tmp_path / name
    return p


# ======================
# Loss / metrics helpers
# ======================

@pytest.mark.skipif(not _has("DiceLoss"), reason="DiceLoss not defined in model.py")
def test_dice_loss_returns_scalar_and_is_finite() -> None:
    criterion = model.DiceLoss()
    logits = torch.randn(4, 1, 64, 64)
    targets = (torch.rand(4, 1, 64, 64) > 0.5).float()

    loss = criterion(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar tensor
    assert torch.isfinite(loss).item()


@pytest.mark.skipif(not _has("bce_dice_loss"), reason="bce_dice_loss not defined in model.py")
def test_bce_dice_loss_returns_scalar_and_is_finite() -> None:
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()

    loss = model.bce_dice_loss(logits, targets, bce_weight=0.7)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


@pytest.mark.skipif(not _has("iou_score"), reason="iou_score not defined in model.py")
def test_iou_score_range_and_extremes() -> None:
    # perfect match: logits strongly positive where target==1 and strongly negative where target==0
    targets = torch.zeros(1, 1, 16, 16)
    targets[..., 4:12, 4:12] = 1.0

    logits = torch.full_like(targets, -10.0)
    logits[targets == 1] = 10.0

    iou = model.iou_score(logits, targets, thr=0.5)
    assert isinstance(iou, float)
    assert 0.0 <= iou <= 1.0
    assert iou > 0.99  # near-perfect

    # complete mismatch: predict all zeros for a non-empty target
    logits2 = torch.full_like(targets, -10.0)
    iou2 = model.iou_score(logits2, targets, thr=0.5)
    assert 0.0 <= iou2 <= 1.0
    assert iou2 < 1e-3


# =========================
# Checkpoint loading helper
# =========================

@pytest.mark.skipif(not _has("load_model"), reason="load_model not defined in model.py")
def test_load_model_roundtrip_state_dict(tmp_path: Path) -> None:
    net = model.UNetSmall(in_ch=3, base_ch=16)
    net.eval()

    # Save a checkpoint containing the raw state_dict
    ckpt_path = _tmp_checkpoint_path(tmp_path, "raw_state.pth")
    torch.save(net.state_dict(), ckpt_path)

    # Load into a fresh instance
    net2 = model.UNetSmall(in_ch=3, base_ch=16)
    net2 = model.load_model(net2, ckpt_path, device="cpu", strict=True)

    # Parameter tensors should match exactly
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), net2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)


@pytest.mark.skipif(not _has("load_model"), reason="load_model not defined in model.py")
def test_load_model_wrapped_checkpoint_and_module_prefix(tmp_path: Path) -> None:
    net = model.UNetSmall(in_ch=3, base_ch=16)

    # Create a fake DataParallel-style state dict with "module." prefix
    sd = {f"module.{k}": v.clone() for k, v in net.state_dict().items()}
    ckpt = {"state_dict": sd}

    ckpt_path = _tmp_checkpoint_path(tmp_path, "wrapped_state.pth")
    torch.save(ckpt, ckpt_path)

    net2 = model.UNetSmall(in_ch=3, base_ch=16)
    net2 = model.load_model(net2, ckpt_path, device="cpu", strict=True)

    # State dict keys should match original (without module.)
    assert set(net2.state_dict().keys()) == set(net.state_dict().keys())


# ===================
# bbox_to_mask helper
# ===================

@pytest.mark.skipif(not _has("bbox_to_mask"), reason="bbox_to_mask not defined in model.py")
def test_bbox_to_mask_basic() -> None:
    import numpy as np

    h, w = 10, 20
    bbox = (2, 3, 7, 8)  # x1,y1,x2,y2
    mask = model.bbox_to_mask(h, w, bbox)

    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert mask.sum() == (8 - 3) * (7 - 2)
    assert mask[3, 2] == 1
    assert mask[0, 0] == 0


# ===============
# OCR crop helper
# ===============

@pytest.mark.skipif(not _has("extract_plate_crop"), reason="extract_plate_crop not defined in model.py")
def test_extract_plate_crop_returns_expected_height() -> None:
    import numpy as np
    import cv2

    # Synthetic image + rectangular mask
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (150, 70), (255, 255, 255), -1)

    mask = np.zeros((120, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 40), (150, 70), 1, -1)

    crop = model.extract_plate_crop(img, mask, out_h=32, max_w=256, use_adaptive=False)

    assert crop is not None
    assert crop.ndim == 2  # grayscale
    assert crop.shape[0] == 32
    assert crop.dtype == np.uint8
