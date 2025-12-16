import torch
from torch.utils.data import Sampler

from typing import Iterator, Optional


# =====================
# Random Subset Sampler
# =====================


class RandomSubsetSampler(Sampler[int]):
    """
    Samples a random subset of indices (without replacement) each time __iter__ is called.
    """

    def __init__(
        self,
        dataset_len: int,
        subset_size: int,
        generator: Optional[torch.Generator] = None,
    ):
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
