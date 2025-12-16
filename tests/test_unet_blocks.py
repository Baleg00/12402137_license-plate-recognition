import pytest
import torch

import model


@pytest.mark.parametrize("B,C,H,W", [(2, 3, 128, 128), (1, 3, 127, 255)])
def test_convblock_shape(B: int, C: int, H: int, W: int) -> None:
    x = torch.randn(B, C, H, W)
    block = model.ConvBlock(in_ch=C, out_ch=32)  # baseline signature
    y = block(x)
    assert y.shape == (B, 32, H, W)


@pytest.mark.parametrize("B,C,H,W", [(2, 32, 128, 128), (1, 32, 127, 255)])
def test_down_halves_spatial(B: int, C: int, H: int, W: int) -> None:
    x = torch.randn(B, C, H, W)
    down = model.Down(in_ch=C, out_ch=64)
    y = down(x)

    # MaxPool2d(2) uses floor division for odd sizes
    assert y.shape[0] == B
    assert y.shape[1] == 64
    assert y.shape[2] == H // 2
    assert y.shape[3] == W // 2


@pytest.mark.parametrize(
    "B, in_ch, skip_ch, out_ch, Hs, Ws",
    [
        # Even dims
        (2, 128, 64, 64, 64, 64),
        # Odd dims: forces padding in Up
        (1, 128, 64, 64, 63, 95),
    ],
)
def test_up_matches_skip_resolution(
    B: int, in_ch: int, skip_ch: int, out_ch: int, Hs: int, Ws: int
) -> None:
    """
    Up(x, skip) should return a feature map at the skip's spatial resolution.
    x is expected to be half-resolution of skip (roughly), but may differ by 1 due to odd dims.
    """
    # create skip at target resolution
    skip = torch.randn(B, skip_ch, Hs, Ws)

    # create x at "bottleneck" resolution ~ half, and with in_ch channels
    # After ConvTranspose2d (stride=2), this will be ~ 2*Hx x 2*Wx; padding should align to skip.
    x = torch.randn(B, in_ch, max(1, Hs // 2), max(1, Ws // 2))

    up = model.Up(in_ch=in_ch, out_ch=out_ch)
    y = up(x, skip)

    assert y.shape[0] == B
    assert y.shape[1] == out_ch
    assert y.shape[2] == Hs
    assert y.shape[3] == Ws


@pytest.mark.parametrize("attention", ["none", "se", "cbam"])
def test_attention_blocks_preserve_shape(attention: str) -> None:
    """
    If your code supports attention selection in ConvBlock (attention='none'|'se'|'cbam'),
    this test verifies shape preservation. If not, it will be skipped.
    """
    if "attention" not in model.ConvBlock.__init__.__code__.co_varnames:
        pytest.skip("ConvBlock does not support attention selection in this version.")

    x = torch.randn(2, 16, 64, 64)
    block = model.ConvBlock(in_ch=16, out_ch=32, attention=attention)
    y = block(x)
    assert y.shape == (2, 32, 64, 64)


def test_seblock_shape_if_present() -> None:
    if not hasattr(model, "SEBlock"):
        pytest.skip("SEBlock not present in model.py")
    x = torch.randn(2, 32, 64, 64)
    se = model.SEBlock(32)
    y = se(x)
    assert y.shape == x.shape


def test_cbam_shape_if_present() -> None:
    if not hasattr(model, "CBAM"):
        pytest.skip("CBAM not present in model.py")
    x = torch.randn(2, 32, 64, 64)
    cbam = model.CBAM(32)
    y = cbam(x)
    assert y.shape == x.shape


def test_dilated_bottleneck_shape_if_present() -> None:
    if not hasattr(model, "DilatedConvBlock"):
        pytest.skip("DilatedConvBlock not present in model.py")
    x = torch.randn(2, 256, 32, 32)
    b = model.DilatedConvBlock(256, 512)
    y = b(x)
    assert y.shape == (2, 512, 32, 32)


@pytest.mark.parametrize("H,W", [(128, 128), (127, 255)])
def test_unetsmall_forward_output_shape(H: int, W: int) -> None:
    """
    End-to-end sanity check: output logits should match input spatial resolution and have 1 channel.
    """
    net = model.UNetSmall(in_ch=3, base_ch=32)
    x = torch.randn(2, 3, H, W)
    y = net(x)
    assert y.shape == (2, 1, H, W)


def test_backward_pass_through_blocks() -> None:
    """
    Ensures gradients flow through the core blocks without errors.
    """
    net = model.UNetSmall(in_ch=3, base_ch=16)  # smaller for speed
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    y = net(x)
    loss = y.mean()
    loss.backward()

    # At least one parameter should have a non-None gradient
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    # And they should not be NaN
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
