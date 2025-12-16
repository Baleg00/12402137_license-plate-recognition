import torch

import re
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Literal

import cv2
import numpy as np
import matplotlib.pyplot as plt

import easyocr

from tqdm import tqdm

from helpers import (
    letterbox_rgb,
    unletterbox_mask,
    overlay_mask,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from training import create_model, load_model


# ================================
# Utilities (pre / postprocessing)
# ================================


def to_model_tensor(rgb_lb: np.ndarray) -> torch.Tensor:
    """RGB uint8 -> normalized float tensor [1,3,H,W]."""
    x = rgb_lb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x).unsqueeze(0).float()


def largest_component(mask01: np.ndarray) -> np.ndarray:
    """Keep largest connected component in a binary mask."""
    m = (mask01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def mask_to_minarearect(mask01: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit minAreaRect to mask; returns 4x2 float32 box points, or None.
    """
    m = (mask01 > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    contour = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:  # tiny noise
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


def order_pts(pts: np.ndarray) -> np.ndarray:
    """Order points as (tl, tr, br, bl)."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_and_crop_plate(
    img_bgr: np.ndarray, mask: np.ndarray
) -> Optional[np.ndarray]:
    """
    Perspective-rectify using minAreaRect from mask, then return BGR crop.
    """
    box = mask_to_minarearect(mask)
    if box is None:
        return None
    box = order_pts(box)

    width = int(max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1])))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(box, dst)
    warp = cv2.warpPerspective(img_bgr, M, (width, height), flags=cv2.INTER_CUBIC)
    return warp


def enhance_for_ocr(plate_bgr: np.ndarray, out_h: int = 48) -> np.ndarray:
    """
    OCR-friendly preprocessing:
    - grayscale
    - CLAHE
    - resize to fixed height
    - mild denoise + sharpen
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # resize to fixed height
    height, width = gray.shape[:2]
    scale = out_h / max(height, 1)
    new_w = max(1, int(round(width * scale)))
    gray = cv2.resize(gray, (new_w, out_h), interpolation=cv2.INTER_CUBIC)

    # mild denoise
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

    # mild unsharp mask
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.3, blur, -0.3, 0)
    return sharp


def normalize_plate_text(s: str) -> str:
    """Uppercase, alphanumeric and last 6 characters."""
    s = re.sub(r"[^A-Z0-9]", "", s.upper())
    if len(s) > 6:
        s = s[-6:]
    return s


def parse_gt_from_stem(stem: str) -> str:
    """Extract ground-truth plate text from filename stem."""
    ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZO"
    ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O"

    fields = stem.split("-")
    if len(fields) != 7:
        return None

    plate_indices = list(map(int, fields[4].split("_")))
    if len(fields) != 7:
        return None

    # ignore first index because it is always a chinese character
    plate = ALPHA[plate_indices[1]] + "".join(
        map(lambda i: ALNUM[i], plate_indices[2:])
    )

    # According to the CCPD documentation: We use O as a sign of "no character"
    # because there is no O in Chinese license plate characters.
    plate.replace("O", "")

    return plate


def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        cur = [i]

        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))

        prev = cur

    return prev[-1]


# ==============================================
# Core pipeline: segment -> crop -> OCR -> score
# ==============================================


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    input_hw: Tuple[int, int] = (512, 512),
    thr: float = 0.5,
) -> np.ndarray:
    """Return binary mask in original image resolution."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    oh, ow = rgb.shape[:2]

    rgb_lb, meta = letterbox_rgb(rgb, input_hw)
    x = to_model_tensor(rgb_lb).to(device, dtype=torch.float32)

    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (probs >= thr).astype(np.uint8)

    pred_orig = unletterbox_mask(pred, (oh, ow), meta)
    pred_orig = largest_component(pred_orig)
    return pred_orig


def easyocr_read(reader: easyocr.Reader, gray_for_ocr: np.ndarray) -> str:
    """Run EasyOCR, return best candidate text (normalized)."""
    # detail=1 returns list of (bbox, text, conf)
    results = reader.readtext(gray_for_ocr, detail=1, paragraph=False)

    if not results:
        return ""

    # pick highest confidence
    best = max(results, key=lambda x: float(x[2]))
    return normalize_plate_text(best[1])


def run_end_to_end_ocr_eval(
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    split_txt: str = "test.txt",
    attention: Literal["none", "se", "cbam"] = "none",
    input_hw: Tuple[int, int] = (512, 512),
    seg_thr: float = 0.5,
    easyocr_langs: List[str] = ["en"],
    shuffle: bool = True,
    max_images: Optional[int] = None,
    strict: bool = True,
    show: bool = False,
) -> Dict[str, float]:
    """
    Iterates images listed by split txt, runs segmentation -> OCR -> compares to GT from stem.
    Prints per-image results and returns summary metrics.
    """
    dataset_root = Path(dataset_root)
    split_path = dataset_root / "splits" / split_txt

    images = [
        (Path(line), Path(line).stem)
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if shuffle:
        random.shuffle(images)

    if max_images is not None:
        images = images[:max_images]

    # Load segmentation model
    model, _, device = create_model(attention=attention)
    model = load_model(model, checkpoint_path, device, strict)
    model.eval()

    # EasyOCR reader
    reader = easyocr.Reader(easyocr_langs, gpu=True)

    n_total = 0
    n_skipped_no_gt = 0
    n_skipped_no_mask = 0

    exact_match = 0
    char_acc_sum = 0.0

    for path, stem in tqdm(images, desc="Evaluating", leave=False):
        img_path = dataset_root / path
        if not img_path.is_file():
            continue

        gt = parse_gt_from_stem(stem)
        if gt is None:
            n_skipped_no_gt += 1
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        # segmentation
        mask = predict_mask(model, bgr, device, input_hw, seg_thr)
        if mask.sum() == 0:
            n_skipped_no_mask += 1
            continue

        # crop + rectify
        plate_bgr = rectify_and_crop_plate(bgr, mask)
        if plate_bgr is None:
            n_skipped_no_mask += 1
            continue

        # post-processing for OCR
        ocr_img = enhance_for_ocr(plate_bgr, out_h=48)

        # OCR
        pred = easyocr_read(reader, ocr_img)

        # scoring
        n_total += 1
        is_em = pred == gt
        exact_match += int(is_em)

        # debug visualization
        if show:
            show_debug_visualization(
                img_bgr=bgr,
                mask=mask,
                plate_bgr=plate_bgr,
                ocr_img=ocr_img,
                stem=stem,
                gt=gt,
                pred=pred,
            )

        # character-level accuracy via normalized Levenshtein
        if len(gt) > 0:
            dist = levenshtein(pred, gt)
            cla = max(0.0, 1.0 - dist / max(len(gt), 1))
        else:
            cla = 0.0
        char_acc_sum += cla

        if show:
            print(f"{path} | GT={gt:s} | OCR={pred:s} | EM={is_em} | CLA={cla:.3f}")

    emr = exact_match / max(n_total, 1)
    cla_mean = char_acc_sum / max(n_total, 1)

    summary = {
        "samples_scored": float(n_total),
        "exact_match_rate": float(emr),
        "char_level_accuracy": float(cla_mean),
        "skipped_no_gt": float(n_skipped_no_gt),
        "skipped_no_mask": float(n_skipped_no_mask),
    }

    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ===================
# Debug Visualization
# ===================


def show_debug_visualization(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    plate_bgr: Optional[np.ndarray],
    ocr_img: Optional[np.ndarray],
    stem: str,
    gt: str,
    pred: str,
) -> None:
    """
    Matplotlib debug view: input, mask, overlay, rectified crop, OCR input.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = overlay_mask(img_rgb, mask)

    plt.figure(figsize=(14, 6))
    plt.suptitle(f"{stem} | GT={gt} | OCR={pred}", fontsize=12)

    plt.subplot(2, 3, 1)
    plt.title("Input")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Predicted mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title(
        "Rectified plate crop"
        if plate_bgr is not None
        else "Rectified plate crop (None)"
    )
    if plate_bgr is not None:
        plt.imshow(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("OCR input" if ocr_img is not None else "OCR input (None)")
    if ocr_img is not None:
        plt.imshow(ocr_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =============
# Main Function
# =============


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end OCR evaluation: segmentation -> crop -> EasyOCR -> compare to GT from stem"
    )

    ap.add_argument("--data", required=True, type=str, help="Dataset root (CCPD)")
    ap.add_argument(
        "--ckpt", required=True, type=str, help="Segmentation checkpoint (.pth/.pt)"
    )
    ap.add_argument(
        "--split",
        default="test.txt",
        choices=["train.txt", "test.txt", "val.txt"],
        help="Which split to evaluate",
    )
    ap.add_argument(
        "--attention",
        default="none",
        choices=["none", "se", "cbam"],
        help="Model variant used for the checkpoint (must match training)",
    )
    ap.add_argument("--input-h", type=int, default=512)
    ap.add_argument("--input-w", type=int, default=512)
    ap.add_argument(
        "--seg-thr",
        type=float,
        default=0.5,
        help="Segmentation threshold on sigmoid(logits)",
    )
    ap.add_argument("--langs", nargs="+", default=["en"], help="EasyOCR languages")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle images")
    ap.add_argument(
        "--max-images", type=int, default=None, help="Limit number of images"
    )
    ap.add_argument("--strict", action="store_true", help="Strict checkpoint loading")
    ap.add_argument(
        "--show", action="store_true", help="Show debug visualizations per image"
    )

    args = ap.parse_args()

    run_end_to_end_ocr_eval(
        dataset_root=Path(args.data),
        checkpoint_path=Path(args.ckpt),
        split_txt=args.split,
        attention=args.attention,
        input_hw=(args.input_h, args.input_w),
        seg_thr=args.seg_thr,
        easyocr_langs=args.langs,
        shuffle=args.shuffle,
        max_images=args.max_images,
        strict=args.strict,
        show=args.show,
    )


if __name__ == "__main__":
    main()
