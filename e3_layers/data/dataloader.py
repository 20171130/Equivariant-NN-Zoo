from typing import List

import torch
import torch.distributed as dist
import math

from .batch import Batch
from .data import Data
from .dataset import CondensedDataset

from absl import flags

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

def getDataIters(config):
  FLAGS = flags.FLAGS
  data_config = config.data_config
  
  # splits the dataset among processes
  rank = dist.get_rank()
  if isinstance(data_config.path, tuple) or isinstance(data_config.path, list):
      gcd = math.gcd(FLAGS.world_size, len(data_config.path))
      start = (rank%gcd) * (len(data_config.path)//gcd)
      end = (rank%gcd + 1)* (len(data_config.path)//gcd)
      data_config.path = data_config.path[start: end]
  dataset = CondensedDataset(**data_config)
  
  # train-val split
  total_n = len(dataset)
  n_train, n_val = data_config.n_train, data_config.n_val
  if isinstance(n_train, float):
      n_train = int(data_config.n_train * total_n)
  if isinstance(n_val, float):
      n_val = int(data_config.n_val * total_n)
  if (n_train + n_val) > total_n:
      raise ValueError(
          "too little data for training and validation. please reduce n_train and n_val"
      )
  if data_config.train_val_split == "random":
      idcs = torch.randperm(total_n)
  elif data_config.train_val_split == "sequential":
      idcs = torch.arange(total_n)
  else:
      raise NotImplementedError(
          f"splitting mode {data_config.train_val_split} not implemented"
      )
  train_idcs = idcs[: n_train]
  val_idcs = idcs[
      n_train : n_train + n_val
  ]
  train_ds = dataset.index_select(train_idcs)
  eval_ds = dataset.index_select(val_idcs)

  # Build data iterators
  loader_rng = (
      torch.Generator()
  )  # used for generating seeds for each dataloader worker process
  loader_rng.manual_seed(FLAGS.seed + dist.get_rank())
  dl_kwargs = dict(
      batch_size=config.batch_size,
      num_workers=FLAGS.dataloader_num_workers,
      pin_memory=True,
      # avoid getting stuck
      timeout=(60 if FLAGS.dataloader_num_workers > 0 else 0),
      generator = loader_rng,
      drop_last = True
  )
  train_dl = DataLoader(
      dataset=train_ds,
      shuffle=True,
      **dl_kwargs,
  )
  eval_dl = DataLoader(
      dataset=eval_ds,
      shuffle=False, 
      **dl_kwargs,
  )
  
  def autoReset(dataloader):
    iterable = iter(dataloader)
    while True:
      try:
        batch = next(iterable)
      except StopIteration:
        iterable = iter(dataloader)
        batch = next(iterable)
      yield batch
    
  return autoReset(train_dl), autoReset(eval_dl)