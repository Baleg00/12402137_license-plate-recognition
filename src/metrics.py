import torch
import torch.nn as nn

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
    