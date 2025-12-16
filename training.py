import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Literal, Tuple, Dict

from tqdm import tqdm

from metrics import DiceLoss
from UNet import UNetSmall
from CCPD import CCPDDataset
from helpers import get_val_transform


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
    attention: Literal["none", "se", "cbam"] = "none"
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.device]:
    model = UNetSmall(in_ch=3, base_ch=32, attention=attention).to(device)
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


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Evaluate loss + IoU on a dataset loader.
    Returns scalar metrics.
    """
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks, _ in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)

        loss = bce_dice_loss(logits, masks)
        iou = iou_score(logits, masks, thr=thr, eps=eps)

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_iou += float(iou) * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "iou": total_iou / max(n, 1),
        "samples": float(n),
    }


def test_checkpoint(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split_txt: str = "test.txt",
    batch_size: int = 8,
    num_workers: int = 4,
    input_hw: Tuple[int, int] = (512, 512),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Load a model from checkpoint and evaluate it on the given split.

    @param checkpoint_path: .pth/.pt checkpoint
    @param dataset_root: CCPD root folder
    @param split: "train.txt", "test.txt" or "val.txt"
    @param batch_size: evaluation batch size
    @param num_workers: dataloader workers
    @param input_hw: (H,W) for val transform
    @param threshold: threshold for IoU binarization

    @return: dict with loss/iou/samples
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_root = Path(dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / loader
    ds = CCPDDataset(
        root=str(dataset_root),
        split_txt=split_txt,
        transform=get_val_transform(target_hw=input_hw),
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model + load weights
    model = UNetSmall(in_ch=3, base_ch=32).to(device)
    model = load_model(model, checkpoint_path, device=device, strict=True)

    # Evaluate
    metrics = evaluate_model(model, loader, device=device, thr=threshold)

    print(
        f"[{split_txt}] checkpoint={checkpoint_path.name} | "
        f"loss={metrics['loss']:.4f} | IoU={metrics['iou']:.4f} | n={int(metrics['samples'])}"
    )
    return metrics
