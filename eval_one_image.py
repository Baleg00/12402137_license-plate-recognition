"""
Evaluate a segmentation checkpoint on a single image and visualize the result.

Usage example:
  python eval_one_image.py
    --ckpt checkpoints/best_cbam.pth
    --image /path/to/img.jpg
    --attention cbam
    --input-h 512
    --input-w 512
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from training import create_model, load_model


def letterbox_rgb(img_rgb: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[float, int, int]]:
    """
    Resize with unchanged aspect ratio and pad to target_hw (H,W).
    Returns padded image and (scale, pad_left, pad_top) for unletterboxing.
    """
    target_height, target_width = target_hw
    height, width = img_rgb.shape[:2]

    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))

    resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, (scale, pad_left, pad_top)


def unletterbox_mask(mask_hw: np.ndarray, original_hw: tuple[int, int], meta: tuple[float, int, int]) -> np.ndarray:
    """
    Undo letterbox on a predicted mask:
    - crop padding
    - resize back to original image size
    """
    scale, pad_left, pad_top = meta
    original_height, original_width = original_hw

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    cropped = mask_hw[pad_top:pad_top + new_height, pad_left:pad_left + new_width]

    out = cv2.resize(cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return out


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Create an overlay highlighting mask region (red) on the RGB image.
    mask must be HxW with values {0,1}.
    """
    overlay = img_rgb.copy()
    red = np.zeros_like(img_rgb)
    red[..., 0] = 255  # red channel

    m = mask.astype(bool)
    overlay[m] = (alpha * red[m] + (1 - alpha) * overlay[m]).astype(np.uint8)

    # Add contour for clarity
    m_u8 = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 0), 2)  # yellow contour

    return overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Checkpoint path (.pth/.pt)")
    ap.add_argument("--image", required=True, type=str, help="Path to input image (.jpg/.png)")
    ap.add_argument("--attention", type=str, default="none", choices=["none", "se", "cbam"])
    ap.add_argument("--base-ch", type=int, default=32, help="Base channels used during training")
    ap.add_argument("--input-h", type=int, default=512)
    ap.add_argument("--input-w", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.5, help="Mask threshold on sigmoid(logits)")
    ap.add_argument("--strict", action="store_true", help="Strict load_state_dict")
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    img_path = Path(args.image)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    oh, ow = rgb.shape[:2]

    # Letterbox to training size
    target_hw = (args.input_h, args.input_w)
    rgb_lb, meta = letterbox_rgb(rgb, target_hw=target_hw)

    # Normalize like ImageNet
    x = rgb_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,3,H,W]

    # Instantiate model and load checkpoint
    
    model, _, device = create_model(attention=args.attention)
    model = load_model(model, args.ckpt, device, args.strict)
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (probs >= args.thr).astype(np.uint8)

    # Map prediction back to original image size
    pred_orig = unletterbox_mask(pred, (oh, ow), meta)

    # Build visualizations
    overlay = overlay_mask(rgb, pred_orig, alpha=0.45)

    # Plate crop preview
    crop_vis = None
    if pred_orig.sum() > 0:
        ys, xs = np.where(pred_orig > 0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        pad = 10
        y1, y2 = max(0, y1 - pad), min(oh, y2 + pad)
        x1, x2 = max(0, x1 - pad), min(ow, x2 + pad)
        crop_vis = rgb[y1:y2, x1:x2]

    # Display
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Predicted mask (thr={args.thr})")
    plt.imshow(pred_orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay (mask highlighted)")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if crop_vis is not None:
        plt.figure(figsize=(6, 3))
        plt.title("Predicted plate crop (bbox from mask)")
        plt.imshow(crop_vis)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
