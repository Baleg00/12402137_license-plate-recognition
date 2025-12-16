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
from sampler import RandomSubsetSampler


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

    for imgs, masks, _ in tqdm(loader, desc="Training", leave=False):
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
) -> tuple[float, float, int]:
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for imgs, masks, _ in tqdm(loader, desc="Validating", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = bce_dice_loss(logits, masks, bce_weight=0.5)
        total_loss += loss.item() * imgs.size(0)
        total_iou += iou_score(logits, masks) * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_iou / n, n


def test_checkpoint(
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split_txt: str = "test.txt",
    batch_size: int = 8,
    num_workers: int = 4,
    input_hw: Tuple[int, int] = (512, 512),
    attention: Literal["none", "se", "cbam"] = "none",
    subset_size: int = 64,
    strict: bool = True,
) -> Dict[str, float]:
    """
    Load a model from checkpoint and evaluate it on the given split.

    @param checkpoint_path: .pth/.pt checkpoint
    @param dataset_root: CCPD root folder
    @param split: "train.txt", "test.txt" or "val.txt"
    @param batch_size: evaluation batch size
    @param num_workers: dataloader workers
    @param input_hw: (H,W) for val transform
    @param attention: attention model
    @param subset_size: test sample subset size
    @param strict: strict checkpoint loading
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_root = Path(dataset_root)

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
        pin_memory=True,
        sampler=RandomSubsetSampler(len(ds), subset_size),
    )

    model, _, device = create_model(attention=attention)
    model = load_model(model, checkpoint_path, device, strict)

    loss, iou, samples = validate(model, loader, device)

    print(
        f"[{split_txt}] checkpoint={checkpoint_path.name} | "
        f"loss={loss:.4f} | IoU={iou:.4f} | n={samples}"
    )
