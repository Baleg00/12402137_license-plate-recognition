"""
License Plate Recognition

Author: Balázs Róna
Matriculation Number: 12402137
"""

import torch
from torch.utils.data import DataLoader

import argparse
from pathlib import Path

from CCPD import CCPDDataset
from helpers import get_train_transform, get_val_transform
from UNet import UNetSmall
from training import train_one_epoch, validate, load_model, test_checkpoint
from sampler import RandomSubsetSampler


# =============
# Main Function
# =============

def build_loaders(
    dataset_root: str | Path,
    input_hw: tuple[int, int],
    batch_size: int,
    num_workers: int,
    subset_size: int,
) -> tuple[DataLoader, DataLoader]:
    dataset_root = Path(dataset_root)

    train_ds = CCPDDataset(
        root=str(dataset_root),
        split_txt="train.txt",
        transform=get_train_transform(target_hw=input_hw),
    )
    test_ds = CCPDDataset(
        root=str(dataset_root),
        split_txt="test.txt",
        transform=get_val_transform(target_hw=input_hw),
    )

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        sampler=RandomSubsetSampler(len(train_ds), subset_size)
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        sampler=RandomSubsetSampler(len(test_ds), subset_size)
    )
    return train_dl, test_dl


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CCPD plate segmentation training/testing CLI")

    sub = parser.add_subparsers(dest="command", required=True)

    # Train from scratch
    p_train = sub.add_parser("train", help="Train a new model from scratch")
    p_train.add_argument("--data", required=True, type=str, help="Dataset root (CCPD)")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--input-h", type=int, default=512)
    p_train.add_argument("--input-w", type=int, default=512)
    p_train.add_argument("--out", type=str, default="checkpoints/model_last.pth")
    p_train.add_argument("--subset-size", type=int, default=64, help="Random sample subset size")
    p_train.add_argument("--attention", type=str, default="none", choices=["none", "se", "cbam"])

    # Resume training from checkpoint
    p_resume = sub.add_parser("resume", help="Load a checkpoint and continue training")
    p_resume.add_argument("--data", required=True, type=str, help="Dataset root (CCPD)")
    p_resume.add_argument("--ckpt", required=True, type=str, help="Checkpoint path (.pth/.pt)")
    p_resume.add_argument("--epochs", type=int, default=20, help="Total epochs to train (not additional)")
    p_resume.add_argument("--batch-size", type=int, default=8)
    p_resume.add_argument("--num-workers", type=int, default=4)
    p_resume.add_argument("--input-h", type=int, default=512)
    p_resume.add_argument("--input-w", type=int, default=512)
    p_resume.add_argument("--out", type=str, default="checkpoints/model_last.pth")
    p_resume.add_argument("--subset-size", type=int, default=64, help="Random sample subset size")
    p_resume.add_argument("--strict", action="store_true", help="Strict checkpoint loading")
    p_resume.add_argument("--attention", type=str, default="none", choices=["none", "se", "cbam"])

    # Test checkpoint
    p_test = sub.add_parser("test", help="Evaluate a checkpoint on the test split")
    p_test.add_argument("--data", required=True, type=str, help="Dataset root (CCPD)")
    p_test.add_argument("--ckpt", required=True, type=str, help="Checkpoint path (.pth/.pt)")
    p_test.add_argument("--batch-size", type=int, default=8)
    p_test.add_argument("--num-workers", type=int, default=4)
    p_test.add_argument("--input-h", type=int, default=512)
    p_test.add_argument("--input-w", type=int, default=512)
    p_test.add_argument("--attention", type=str, default="none", choices=["none", "se", "cbam"])
    p_test.add_argument("--subset-size", type=int, default=64, help="Random sample subset size")
    p_test.add_argument("--strict", action="store_true", help="Strict checkpoint loading")

    args = parser.parse_args()

    input_hw = (args.input_h, args.input_w)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loaders when needed
    if args.command in ("train", "resume"):
        train_dl, test_dl = build_loaders(
            dataset_root=args.data,
            input_hw=input_hw,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size
        )

    # Train from scratch
    if args.command == "train":
        model = UNetSmall(in_ch=3, base_ch=32, attention=args.attention).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        for epoch in range(args.epochs):
            tr_loss = train_one_epoch(model, train_dl, optimizer, device)
            te_loss, te_iou = validate(model, test_dl, device)
            print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | test {te_loss:.4f} | IoU {te_iou:.4f}")

            save_checkpoint(args.out, model, optimizer, epoch)

        return

    # Resume training from checkpoint
    if args.command == "resume":
        model = UNetSmall(in_ch=3, base_ch=32, attention=args.attention).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        ckpt = torch.load(args.ckpt, map_location="cpu")
        model = load_model(model, args.ckpt, device=device, strict=args.strict)

        start_epoch = 0
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("Warning: could not load optimizer state (continuing with fresh optimizer).")
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1

        for epoch in range(start_epoch, args.epochs):
            tr_loss = train_one_epoch(model, train_dl, optimizer, device)
            te_loss, te_iou = validate(model, test_dl, device)
            print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | test {te_loss:.4f} | IoU {te_iou:.4f}")

            save_checkpoint(args.out, model, optimizer, epoch)

        return

    # Test checkpoint
    if args.command == "test":
        test_checkpoint(
            checkpoint_path=args.ckpt,
            dataset_root=args.data,
            split_txt="test.txt",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_hw=input_hw,
            attention=args.attention,
            subset_size=args.subset_size,
            strict=args.strict,
        )

        return


if __name__ == "__main__":
    main()
