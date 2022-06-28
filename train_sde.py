import gc
import io
import os
import time
import logging
from tqdm import tqdm, trange

import numpy as np
import tensorflow as tf
import torch
import wandb
from torchvision.utils import make_grid, save_image

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import sde_utils as losses
import sde_utils as sde_lib
from sde_utils import getScaler
import sde_sampling as sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets as datasets 
import likelihood as likelihood
from utils import save_checkpoint, restore_checkpoint

from absl import flags, app
from ml_collections.config_flags import config_flags

from e3_layers.utils import build, saveMol, pruneArgs
from e3_layers import configs
from e3_layers.data import CondensedDataset, DataLoader

import pdb

FLAGS = flags.FLAGS

def train(sde_config, e3_config):
  """Runs the training pipeline.

  Args:
    sde_config: the config for sde_score_pytorch https://github.com/yang-song/score_sde_pytorch
    e3_config: the config for e3_layers https://github.com/20171130/Equivariant-NN-Zoo
  """
  workdir = FLAGS.workdir

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)

  # Initialize model.
  score_model = build(e3_config.model_config).to(sde_config.device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=sde_config.model.ema_rate)
  
  optim = getattr(torch.optim, e3_config.optimizer_name)
  kwargs = pruneArgs(prefix="optimizer", **e3_config)
  kwargs.pop('name')
  optimizer = optim(
      params=score_model.parameters(), lr=e3_config.learning_rate, **kwargs
  )
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, sde_config.device)
  initial_step = int(state['step'])

  # Setup SDEs
  if sde_config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=sde_config.model.beta_min, beta_max=sde_config.model.beta_max, N=sde_config.model.num_scales)
    sampling_eps = 1e-3
  elif sde_config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=sde_config.model.beta_min, beta_max=sde_config.model.beta_max, N=sde_config.model.num_scales)
    sampling_eps = 1e-3
  elif sde_config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=sde_config.model.sigma_min, sigma_max=sde_config.model.sigma_max, N=sde_config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {sde_config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  continuous = sde_config.training.continuous
  reduce_mean = sde_config.training.reduce_mean
  likelihood_weighting = sde_config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimizer=optimizer,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, grad_clid_norm=e3_config.grad_clid_norm)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimizer=optimizer,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Build data iterators
  train_iter, eval_iter = getDataLoaders(e3_config)
  
  # Create data normalizer and its inverse
  std = getattr(e3_config.data_config, 'std', 1)
  scaler = getScaler(1/std)
  inverse_scaler = getScaler(std)
  
  # Building sampling functions
  if sde_config.training.snapshot_sampling:
    sampling_fn = sampling.get_sampling_fn(sde_config, sde, inverse_scaler, sampling_eps)

  num_train_steps = sde_config.training.n_iters
    
  scheduler = getattr(torch.optim.lr_scheduler, e3_config.lr_scheduler_name)
  kwargs = pruneArgs(prefix="lr_scheduler", **e3_config)
  kwargs.pop('name')
  lr_sched = scheduler(optimizer=optimizer, **kwargs)
          
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
    
  pbar = trange(initial_step, num_train_steps + 1)
  pbar.set_description(f'{FLAGS.name}')
  
  for step in pbar:
    try:
      batch = next(train_iter).to(sde_config.device)
    except StopIteration:
      train_iter, _ = getDataLoaders(e3_config)
      batch = next(train_iter).to(sde_config.device)
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % FLAGS.log_period == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      wandb.log(dict(loss = loss.item(), optim_step = step))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % FLAGS.save_period == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % FLAGS.eval_period == 0:
      try:
        eval_batch = next(eval_iter).to(sde_config.device)
      except StopIteration:
        _, eval_iter = getDataLoaders(e3_config)
        eval_batch = next(eval_iter).to(sde_config.device)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      wandb.log(dict(eval_loss = eval_loss.item(), lr = optimizer.param_groups[0]["lr"], optim_step = step))
      lr_sched.step(metrics=eval_loss.item())
      
    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % FLAGS.save_period == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // FLAGS.save_period 
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if sde_config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        
        saveMol(inverse_scaler(batch), workdir=FLAGS.workdir, filename='ground_truth.gro')
        wandb.log({'ground_truth': wandb.Molecule(os.path.join(FLAGS.workdir, 'ground_truth.gro'))})

        sample, n = sampling_fn(score_model, batch)

        saveMol(inverse_scaler(sample), workdir=FLAGS.workdir, filename='sample.gro')
        wandb.log({'sample': wandb.Molecule(os.path.join(FLAGS.workdir, 'sample.gro'))})
          
          
def getDataLoaders(e3_config):
  data_config = e3_config.data_config
  dataset = CondensedDataset(**data_config)
  total_n = len(dataset)
  if (data_config.n_train + data_config.n_val) > total_n:
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

  train_idcs = idcs[: data_config.n_train]
  val_idcs = idcs[
      data_config.n_train : data_config.n_train + data_config.n_val
  ]
  train_ds = dataset.index_select(train_idcs)
  eval_ds = dataset.index_select(val_idcs)

  dl_kwargs = dict(
      batch_size=e3_config.batch_size,
      num_workers=FLAGS.dataloader_num_workers,
      pin_memory=True,
      # avoid getting stuck
      timeout=(10 if FLAGS.dataloader_num_workers > 0 else 0)
  )
  train_dl = DataLoader(
      dataset=dataset,
      shuffle=True,
      **dl_kwargs,
  )
  eval_dl = DataLoader(
      dataset=dataset,
      shuffle=False, 
      **dl_kwargs,
  )
  train_iter, eval_iter = iter(train_dl), iter(eval_dl)

  return train_iter, eval_iter
        
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "sde_config", None, "Training sde_configuration.", lock_config=True)
flags.DEFINE_string("workdir", 'results', "Work directory.")

flags.DEFINE_string("e3_config", None, "The name of the config.")
flags.DEFINE_string("config_spec", '', "Config specification.")
flags.DEFINE_string("name", "default", "Name of the experiment.")
flags.DEFINE_integer("seed", None, "The RNG seed.")
flags.DEFINE_integer(
    "dataloader_num_workers", 4, "Number of workers per training process."
)
flags.DEFINE_boolean("wandb", False, "If logging with wandb.")
flags.DEFINE_string("wandb_project", None, "The name of the wandb project.")
flags.DEFINE_string("verbose", "INFO", "Logging verbosity.")
flags.DEFINE_integer("log_period", 100, "Number of steps.")
flags.DEFINE_integer("eval_period", 100, "Number of steps.")
flags.DEFINE_integer("save_period", 2000, "Number of steps.")
flags.mark_flags_as_required(["sde_config", "e3_config"])


def main(argv):
  # prevent tf from taking up all GPU memory, causing torch to raise OOM
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      raise e
      
  FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.name)
  # Create the working directory
  tf.io.gfile.makedirs(FLAGS.workdir)
  # Set logger so that it outputs to both console and file
  # Make logging work for both disk and Google Cloud Storage
  gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel(FLAGS.verbose)
  # Run the training pipeline

  config_name = FLAGS.e3_config
  e3_config = getattr(configs, config_name, None)
  assert not e3_config is None, f"Config {config_name} not found."
  e3_config = e3_config(FLAGS.config_spec)
  
  FLAGS.sde_config.training.batch_size = e3_config.batch_size
  
  config_dict = {'e3': e3_config.to_dict(), 'sde': FLAGS.sde_config.to_dict()}
  run_id = wandb.util.generate_id()
  if FLAGS.wandb:
    mode = 'online'
  else:
    mode = 'disabled'
  wandb.init(
      project=FLAGS.wandb_project,
      config=config_dict,
      mode = mode,
      name=f"{FLAGS.name}_{FLAGS.seed}",
      dir = FLAGS.workdir,
      resume="allow",
      id=run_id,
      settings=wandb.Settings(),
  )

  if FLAGS.seed is not None:
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
          
  train(FLAGS.sde_config, e3_config)


if __name__ == "__main__":
  app.run(main)