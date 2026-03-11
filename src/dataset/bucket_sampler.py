"""Mini-batch sampling utilities."""

from __future__ import annotations

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """Groups similarly sized graphs into the same mini-batch."""

    def __init__(
        self,
        sizes: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.sizes = list(sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        ordered = sorted(range(len(self.sizes)), key=lambda index: self.sizes[index])
        batches = [
            ordered[start : start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        batch_count, remainder = divmod(len(self.sizes), self.batch_size)
        if remainder and not self.drop_last:
            return batch_count + 1
        return batch_count

