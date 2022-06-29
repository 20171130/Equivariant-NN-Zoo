import gc
import io
import os
import time
import logging
from tqdm import tqdm, trange
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import sde_utils as losses
import sde_utils as sde_lib
from sde_utils import getScaler
import sde_sampling as sampling

from models.ema import ExponentialMovingAverage
import likelihood as likelihood
from utils import save_checkpoint, restore_checkpoint

from absl import flags, app
from ml_collections.config_flags import config_flags

from e3_layers.utils import build, pruneArgs
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeVector, getDataIters

import pdb

FLAGS = flags.FLAGS

def train(e3_config, sde_config):
  """Runs the training pipeline.

  Args:
    sde_config: the config for sde_score_pytorch https://github.com/yang-song/score_sde_pytorch
    e3_config: the config for e3_layers https://github.com/20171130/Equivariant-NN-Zoo
  """
  workdir = FLAGS.workdir
  saveMol = e3_config.saveMol
  device = torch.device(dist.get_rank())
  
  if dist.get_rank() == 0:
    # Create checkpoints directory
    checkpoint_dir = os.path.join(FLAGS.workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(FLAGS.workdir, "checkpoints-meta", "checkpoint.pth")
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(os.path.dirname(checkpoint_meta_dir)).mkdir(exist_ok=True)
  
  # Initialize model.
  score_model = build(e3_config.model_config).to(device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=sde_config.model.ema_rate)
  optim = getattr(torch.optim, e3_config.optimizer_name)
  kwargs = pruneArgs(prefix="optimizer", **e3_config)
  kwargs.pop('name')
  optimizer = optim(
      params=score_model.parameters(), lr=e3_config.learning_rate, **kwargs
  )
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, sde_config.device)
  initial_step = int(state['step'])
  score_model = DDP(score_model)
    
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
                                     likelihood_weighting=likelihood_weighting,
                                     grad_clid_norm=e3_config.grad_clid_norm,
                                     grad_acc = e3_config.grad_acc)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimizer=optimizer,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  train_iter, eval_iter = getDataIters(e3_config)
  
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
  
  loss_lst = []
  eval_loss_lst = []
  for step in pbar:
    batch = next(train_iter).to(device)
    batch = scaler(batch)
    # Execute one training step
    
    loss = train_step_fn(state, batch).item()
    loss_lst.append(loss)
    if step % FLAGS.log_period == 0 and step>0:
      logging.info("step: %d, training_loss: %.5e" % (step, sum(loss_lst)/len(loss_lst)))
      wandb.log(dict(loss = sum(loss_lst)/len(loss_lst), optim_step = step))
      loss_lst = []

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % FLAGS.save_period == 0 and dist.get_rank() == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % FLAGS.eval_period == 0:
      eval_batch = next(eval_iter).to(device)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch).item()
      eval_loss_lst.append(eval_loss)
      
    # Save a checkpoint periodically and generate samples if needed
    if (step != 0 and step % FLAGS.save_period == 0 or step == num_train_steps) and dist.get_rank()==0:
      if len(eval_loss_lst)> 0:
        eval_loss_mean = sum(eval_loss_lst)/len(eval_loss_lst)
      else:
        eval_loss_mean = float('inf')
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss_mean))
      lr_sched.step(metrics=eval_loss_mean)
      eval_loss_lst = []
      wandb.log(dict(eval_loss = eval_loss_mean, lr = optimizer.param_groups[0]["lr"], optim_step = step))
      
      # Save the checkpoint.
      save_step = step // FLAGS.save_period 
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if sde_config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        ema.restore(score_model.parameters())
        sample_dir = os.path.join(workdir, "samples")
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
        
        filenmae = saveMol(inverse_scaler(batch), workdir=FLAGS.workdir, filename='ground_truth')
        wandb.log({'ground_truth': wandb.Molecule(filenmae)})

        n_samples = 5
        lst = [batch[0] for i in range(n_samples)]
        batch = Batch.from_data_list(lst, batch.attrs)
        samples_batch, n = sampling_fn(score_model, batch)
        samples = [samples_batch[i] for i in range(len(samples_batch))]
        
        batch = computeEdgeVector(batch)
        min_loss = 9999
        argmin = 0
        sum_loss = 0
        for i, sample in enumerate(samples):
          loss = (batch[0]['edge_length'] - sample['edge_length'])**2
          loss = loss.mean().item()
          sum_loss += loss
          if loss < min_loss:
            min_loss = loss
            argmin = i

        filename = f'{step}_{sum_loss/n_samples}'
        filenmae = saveMol(samples_batch, idx=i, workdir=FLAGS.workdir, filename=filename)
        wandb.log({'sample': wandb.Molecule(filenmae)})
        
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "sde_config", None, "Training sde_configuration.", lock_config=True)
flags.DEFINE_string("workdir", 'results', "Work directory.")

flags.DEFINE_string("e3_config", None, "The name of the config.")
flags.DEFINE_string("config_spec", '', "Config specification.")
flags.DEFINE_string("name", "default", "Name of the experiment.")
flags.DEFINE_integer("seed", 0, "The RNG seed.")
flags.DEFINE_integer(
    "dataloader_num_workers", 4, "Number of workers per training process."
)
flags.DEFINE_boolean("wandb", False, "If logging with wandb.")
flags.DEFINE_string("wandb_project", None, "The name of the wandb project.")
flags.DEFINE_string("verbose", "INFO", "Logging verbosity.")
flags.DEFINE_integer("log_period", 100, "Number of training steps.")
flags.DEFINE_integer("eval_period", 20, "Number of training steps.")
flags.DEFINE_integer("save_period", 2000, "Number of training steps.")

flags.DEFINE_integer("world_size", 1, "Number of processes.")
flags.DEFINE_string("master_addr", "127.0.0.1", "The address to use.")
flags.DEFINE_string("master_port", "10000", "The port to use.")
flags.mark_flags_as_required(["sde_config", "e3_config"])


def main(rank, e3_config):
  logger = logging.getLogger()
  if rank == 0:
      # Set logger so that it outputs to both console and file
      # Make logging work for both disk and Google Cloud Storage
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
      handler = logging.StreamHandler(gfile_stream)
      formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
      handler.setFormatter(formatter)
      for handler in logger.handlers:
        handler.setFormatter(formatter)
      logger.addHandler(handler)
      logger.setLevel(getattr(logging, FLAGS.verbose))
      # Run the training pipeline
  else:
      logger.setLevel(logging.WARNING)
      
  config_dict = {'e3': e3_config.to_dict(), 'sde': FLAGS.sde_config.to_dict()}
  run_id = wandb.util.generate_id()
  if FLAGS.wandb and rank == 0:
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

  torch.manual_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  
  mp.set_start_method("fork", force=True)
  dist.init_process_group("nccl", rank=rank, world_size=FLAGS.world_size)
          
  train(e3_config, FLAGS.sde_config)


def launch_mp(argv):
    world_size = FLAGS.world_size
    os.environ["MASTER_ADDR"] = FLAGS.master_addr
    os.environ["MASTER_PORT"] = FLAGS.master_port
    FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.name)
    # Create the working directory
    Path(FLAGS.workdir).mkdir(exist_ok=True)

    config_name = FLAGS.e3_config
    e3_config = getattr(configs, config_name, None)
    assert not e3_config is None, f"Config {config_name} not found."
    e3_config = e3_config(FLAGS.config_spec)

    FLAGS.sde_config.training.batch_size = e3_config.batch_size
  
      
    if world_size == 1:
        main(0, e3_config)
    else:
        mp.spawn(main, args=(e3_config), nprocs=config.world_size, join=True)

if __name__ == "__main__":
    app.run(launch_mp)