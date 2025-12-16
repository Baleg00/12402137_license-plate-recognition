import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ================
# Helper Functions
# ================

def letterbox(img, mask, target_hw=(512, 512), border_value=(114, 114, 114)):
    """
    Letterbox/pad image to target size.

    @param img: The image to resize/pad.
    @param mask: The mask to resize/pad.
    @param target_hw: Target image (height, width) tuple.
    @param border_value: Padding border color.
    @return: (img_padded, mask_padded) - Tuple of padded image and mask.
    """

    height, width = img.shape[:2]
    target_height, target_width = target_hw
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    img_resized = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    # pad to target size
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return img_padded, mask_padded


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


def bbox_to_mask(height, width, bbox):
    """
    bbox = (x_min, y_min, x_max, y_max) in pixel coords
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2, x1:x2] = 1
    return mask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(target_hw=(512, 512)):
    """
    Create random transformations to augment training images.

    @param target_hw: Target image (height, width) tuple.
    @return: Albumentations composition of random transformations.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max(target_hw), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=target_hw[0],
                min_width=target_hw[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                fill_mask=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-8, 8),
                shear=(-5, 5),
                translate_percent=(0.02, 0.02),
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5
            ),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.15),
            A.ImageCompression(quality_range=(60, 100), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transform(target_hw=(512, 512)):
    """
    Create random transformations to augment validation images.

    @param target_hw: Target image (height, width) tuple.
    @return: Albumentations composition of random transformations.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max(target_hw), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=target_hw[0],
                min_width=target_hw[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                fill_mask=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def extract_plate_crop(img_bgr, mask_bin, out_h=32, max_w=256, use_adaptive=False):
    """
    OCR crop normalization helper.

    @param img_bgr: Original BGR image (uint8).
    @param mask_bin: Binary mask (uint8 {0, 1}) same height and width.
    @param out_h: Output image height.
    @param max_w: Maximum width of output image.
    @param use_adaptive: Use adaptive threshold.
    @return: A normalized plate crop for OCR.
    """

    mask = (mask_bin > 0).astype(np.uint8)

    if mask.sum() == 0:
        return None  # no plate found

    # largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return None

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cc = (labels == largest).astype(np.uint8)

    # get contour and minAreaRect for rough rectification
    contours, _ = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area_contour = max(contours, key=cv2.contourArea)
    contour_rect = cv2.minAreaRect(max_area_contour)
    contour_box = cv2.boxPoints(contour_rect).astype(np.float32)

    # order box points (tl, tr, br, bl)
    s = contour_box.sum(axis=1)
    diff = np.diff(contour_box, axis=1).reshape(-1)
    top_left = contour_box[np.argmin(s)]
    bottom_right = contour_box[np.argmax(s)]
    top_right = contour_box[np.argmin(diff)]
    bottom_left = contour_box[np.argmax(diff)]

    contour_box = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )

    # width/height for warp (keep aspect roughly plate-like)
    width = int(
        max(
            np.linalg.norm(contour_box[1] - contour_box[0]),
            np.linalg.norm(contour_box[2] - contour_box[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(contour_box[3] - contour_box[0]),
            np.linalg.norm(contour_box[2] - contour_box[1]),
        )
    )
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(contour_box, dst)
    warp = cv2.warpPerspective(img_bgr, M, (width, height), flags=cv2.INTER_CUBIC)

    # grayscale + contrast normalization
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    # CLAHE helps a lot on low-light/overexposed plates
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    # resize to OCR-friendly height
    scale = out_h / norm.shape[0]
    new_w = min(int(round(norm.shape[1] * scale)), max_w)
    ocr_img = cv2.resize(norm, (new_w, out_h), interpolation=cv2.INTER_CUBIC)

    if use_adaptive:
        ocr_img = cv2.adaptiveThreshold(
            ocr_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    return ocr_img  # uint8, H = out_h, W <= max_w
