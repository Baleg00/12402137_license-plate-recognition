"""
License Plate Recognition

Author: Balázs Róna
Matriculation Number: 12402137


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import json
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


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


# ============
# CCPD Dataset
# ============

class CCPDDataset(Dataset):
    def __init__(self, root, split_txt, transform=None):
        """
        CCPD dataset loader.

        Directory structure:
        root/
            splits/
                train.txt
                val.txt
                test.txt
            ccpd_base/
                *.jpg
            ...

        @param root: Dataset root directory.
        @param split_txt: One of {"train.txt", "val.txt", "test.txt"}.
        @param transform: Albumentations transform (image + optional bboxes).
        """
        self.root = Path(root)
        self.transform = transform

        with open(self.root / "splits" / split_txt) as f:
            self.img_rel_paths = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_rel_paths)

    def _parse_filename(self, name):
        """
        Parse CCPD filename annotations.
        """
        stem = Path(name).stem
        fields = stem.split("-")
        assert len(fields) == 7, f"Unexpected filename format: {name}"

        # Area ratio
        area = float(fields[0])

        # Tilt degrees
        tilt_h, tilt_v = map(float, fields[1].split("_"))

        # Bounding box: left-up & right-bottom
        lu, rb = fields[2].split("_")
        x1, y1 = map(int, lu.split("&"))
        x2, y2 = map(int, rb.split("&"))
        bbox = [x1, y1, x2, y2]

        # Four vertices (starting from right-bottom, clockwise)
        vertices = []
        for v in fields[3].split("_"):
            x, y = map(int, v.split("&"))
            vertices.append([x, y])
        vertices = np.array(vertices, dtype=np.float32)

        # License plate number (encoded indices)
        plate_indices = list(map(int, fields[4].split("_")))

        # Brightness and blurriness
        brightness = int(fields[5])
        blurriness = int(fields[6])

        return {
            "area": area,
            "tilt": (tilt_h, tilt_v),
            "bbox": bbox,
            "vertices": vertices,
            "plate_indices": plate_indices,
            "brightness": brightness,
            "blurriness": blurriness,
        }

    def __getitem__(self, idx):
        rel_path = self.img_rel_paths[idx]
        img_path = self.root / rel_path
        stem = Path(rel_path).stem

        # Read image (RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Parse annotations from filename
        ann = self._parse_filename(img_path.name)

        # Build binary mask from polygon (four vertices)
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = ann["vertices"]  # shape [4,2], float32
        poly_i = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [poly_i], 1)

        # Apply transform consistently (image + mask)
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img_out = out["image"]
            mask_out = out["mask"]

            # Ensure [1, H, W] float mask
            if isinstance(mask_out, torch.Tensor):
                mask_out = mask_out.unsqueeze(0).float()
            else:
                mask_out = mask_out[None, ...].astype(np.float32)

            return img_out, mask_out, stem

        # No transform: return numpy arrays
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = mask[None, ...].astype(np.float32)

        return img, mask, stem


# =====================
# U-Net Building Blocks
# =====================

class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale with maxpool then double conv"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscale then double conv. Uses transposed conv for upsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)  # in_ch = skip(ch) + up(ch)

    def forward(self, x, skip):
        x = self.up(x)

        # pad if needed (handles odd dims)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)

        x = F.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNetSmall(nn.Module):
    """
    Lightweight U-Net for binary segmentation (1 class: plate vs background).
    Input: 3xHxW, Output: 1xHxW logits (use with BCEWithLogits).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()

        # Encoder
        self.inc = ConvBlock(in_ch, base_ch)  # 3 -> 32
        self.down1 = Down(base_ch, base_ch * 2)  # 32 -> 64
        self.down2 = Down(base_ch * 2, base_ch * 4)  # 64 -> 128
        self.down3 = Down(base_ch * 4, base_ch * 8)  # 128 -> 256

        # Optional extra depth for larger model
        # self.down4 = Down(base_ch * 8, base_ch * 16)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        # Decoder
        self.up3 = Up(base_ch * 16, base_ch * 8)  # (256 + 256) -> 256
        self.up2 = Up(base_ch * 8, base_ch * 4)  # (128 + 128) -> 128
        self.up1 = Up(base_ch * 4, base_ch * 2)  # (64 + 64)   -> 64
        self.up0 = Up(base_ch * 2, base_ch)  # (32 + 32)   -> 32

        # Head
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)  # H
        x1 = self.down1(x0)  # H/2
        x2 = self.down2(x1)  # H/4
        x3 = self.down3(x2)  # H/8

        xb = self.bottleneck(x3)

        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        logits = self.outc(y0)  # raw logits

        return logits


# ==================================
# Loss & Simple Metrics (Dice + BCE)
# ==================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1, H, W], targets: [B, 1, H, W] in {0, 1}
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(2, 3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        dice = 1.0 - (num / den)
        return dice.mean()


# =====================
# Random Subset Sampler
# =====================

class RandomSubsetSampler(Sampler[int]):
    """
    Samples a random subset of indices (without replacement) each time __iter__ is called.
    """
    def __init__(self, dataset_len: int, subset_size: int, generator: Optional[torch.Generator] = None):
        if subset_size <= 0:
            raise ValueError("subset_size must be > 0")
        if subset_size > dataset_len:
            raise ValueError("subset_size cannot exceed dataset_len")
        self.dataset_len = dataset_len
        self.subset_size = subset_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # new random subset each epoch
        perm = torch.randperm(self.dataset_len, generator=self.generator)
        subset = perm[: self.subset_size].tolist()
        return iter(subset)

    def __len__(self) -> int:
        return self.subset_size


# ==================
# Training Functions
# ==================

def bce_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = DiceLoss()(logits, targets)
    return bce_weight * bce + (1 - bce_weight) * dice


@torch.no_grad()
def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3)) + eps
    iou = (inter + eps) / union
    return iou.mean().item()


def create_model(
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.device]:
    model = UNetSmall(in_ch=3, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    return model, optimizer, device


def load_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load model weights from a checkpoint file.

    @param model: Instantiated model (architecture must match checkpoint).
    @param checkpoint_path: Path to .pt or .pth file.
    @param device: Target device ("cpu", "cuda", torch.device). If None, auto-detect.
    @param strict: Whether to strictly enforce that the keys in state_dict match.

    @return: Model with loaded weights, moved to the target device.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load checkpoint to CPU first (safe and portable)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoints
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel / DDP prefixes if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for imgs, masks, _ in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = bce_dice_loss(logits, masks, bce_weight=0.5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for imgs, masks, _ in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = bce_dice_loss(logits, masks, bce_weight=0.5)
        total_loss += loss.item() * imgs.size(0)
        total_iou += iou_score(logits, masks) * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_iou / n


# =============
# Main Function
# =============

def main():
    CCPD_ROOT = "PATH-TO-ROOT"

    train_ds = CCPDDataset(
        root=CCPD_ROOT,
        split_txt="train.txt",
        transform=get_train_transform(target_hw=(512, 512)),
    )

    test_ds = CCPDDataset(
        root=CCPD_ROOT,
        split_txt="test.txt",
        transform=get_train_transform(target_hw=(512, 512)),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=10,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=RandomSubsetSampler(len(train_ds), subset_size=300)
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=10,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=RandomSubsetSampler(len(test_ds), subset_size=50)
    )

    model, optim, device = create_model()
    # model = load_model(model, "PATH TO MODEL SAVE")

    print(f"Device: {device}")

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_dl, optim, device)
        test_loss, test_iou = validate(model, test_dl, device)

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | test {test_loss:.4f} | IoU {test_iou:.4f}")

        model_path = f"./model-{int(time.time())}-{int(train_loss * 1e4)}-{int(test_loss * 1e4)}-{int(test_iou * 1e4)}.pt"

        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
