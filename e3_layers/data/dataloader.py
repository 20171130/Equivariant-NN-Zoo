from typing import List

import torch

from .batch import Batch
from .data import Data


class Collater(object):
    def __init__(self, attrs={}):
        self.attrs = attrs

    @classmethod
    def for_dataset(cls, dataset):
        return cls(attrs=dataset.attrs)

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        out = Batch.from_data_list(batch, attrs=self.attrs)
        return out

    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater.for_dataset(dataset),
            **kwargs,
        )
