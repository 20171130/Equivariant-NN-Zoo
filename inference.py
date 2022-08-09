from absl import app
from absl import flags

import os
from pathlib import Path
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist
from e3_layers.utils import build, setSeed, _countParameters
from e3_layers import configs
from e3_layers.data import CondensedDataset, DataLoader, Batch, getDataIters
from ml_collections.config_flags import config_flags


FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "The name of the config.")
flags.DEFINE_string("config_spec", 'eval', "Config specification, the argument of get_config().")
flags.DEFINE_string("output_path", None, "Path to the output file to create.")
flags.DEFINE_string("name", "default", "Name of the experiment.")
flags.DEFINE_string("model_path", None, "The name of the model checkpoint.")
flags.DEFINE_list("output_keys", [], "The output keys to save.")
flags.DEFINE_integer("seed", 0, "The RNG seed.")
flags.DEFINE_integer(
    "dataloader_num_workers", 4, "Number of workers per training process."
)
flags.DEFINE_string("verbose", "INFO", "Logging verbosity.")
flags.DEFINE_string("master_addr", "127.0.0.1", "The address to use.")
flags.DEFINE_string("master_port", "10000", "The port to use.")
config_flags.DEFINE_config_file(
  "sde_config", None, "Training sde_configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])


def regression(argv):
    config_name = FLAGS.config
    config = getattr(configs, config_name, None)
    assert not config is None, f"Config {config_name} not found."
    config = config(FLAGS.config_spec)
    
    model = build(config.model_config)
    device = torch.device('cuda')
    model.to(device=device)
    if FLAGS.model_path:
        state_dict = torch.load(FLAGS.model_path, map_location=device)
        model_state_dict = {}
        for key, value in state_dict.items():
            if key[:7] == 'module.': # remove DDP wrappers
                key = key[7:]
            model_state_dict[key] = value
            model_state_dict.pop(key)
        model.load_state_dict(model_state_dict)
    
    data_config = config.data_config
    dataset = CondensedDataset(**data_config)
    dl_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=FLAGS.dataloader_num_workers,
        pin_memory=(device != torch.device("cpu")),
        # avoid getting stuck
        timeout=(10 if FLAGS.dataloader_num_workers > 0 else 0)
    )
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False, 
        **dl_kwargs,
    )
    
    model.eval()
    lst = []
    for batch in tqdm(iter(dataloader)):
        batch = batch.to(device)
        out = model(batch.clone())
        dic = {key: out[key] for key in FLAGS.output_keys}
        lst.append(dic)
    result = Batch.from_data_list(lst, attrs = batch.attrs)
    result.dumpHDF5(FLAGS.output_path)
    
def diffusion(e3_config, FLAGS):
  from models.ema import ExponentialMovingAverage
  import e3_layers.run.sde_utils as losses
  import e3_layers.run.sde_utils as sde_lib
  import e3_layers.run.sde_sampling as sampling
  """Runs the training pipeline.

  Args:
    sde_config: the config for sde_score_pytorch https://github.com/yang-song/score_sde_pytorch
    e3_config: the config for e3_layers https://github.com/20171130/Equivariant-NN-Zoo
  """
  sde_config = FLAGS.sde_config
  saveMol = e3_config.saveMol
  device = torch.device(0)
  
  # Initialize model.
  model = build(e3_config.model_config).to(device)
  setSeed(FLAGS.seed) # must reset seed after JIT to keep it the same across processes
  logging.info(f'Number of parameters {_countParameters(model)}.')
  
  if FLAGS.model_path is not None:
    state_dict = torch.load(FLAGS.model_path, map_location=device)
    state_dict = state_dict['model']
    model_state_dict = {}
    for key, value in state_dict.items():
      if key[:7] == 'module.': # remove DDP wrappers
        key = key[7:]
      model_state_dict[key] = value
    model.load_state_dict(model_state_dict)
    logging.info(f"Resumed from checkpoint {FLAGS.model_path}.")

  # Setup SDEs
  sde = sde_lib.VPSDE(beta_min=sde_config.model.beta_min, beta_max=sde_config.model.beta_max, N=sde_config.model.num_scales, diffusion_keys=e3_config.diffusion_keys)
  sampling_eps = 1e-3

  # Build one-step training and evaluation functions
  continuous = sde_config.training.continuous
  likelihood_weighting = sde_config.training.likelihood_weighting
  e3_config.batch_size = 1
  iterator = getDataIters(e3_config, test=True)
  
  # Create data normalizer and its inverse
  scaler = e3_config.data_config.scaler
  inverse_scaler = e3_config.data_config.inverse_scaler
  
  # Building sampling functions
  sampling_fn = sampling.get_sampling_fn(sde_config, sde, inverse_scaler, sampling_eps)

  for step, batch in enumerate(iterator):
    batch = scaler(batch).to(device)
    n_samples = 1
    lst = [batch[0] for i in range(n_samples)]
    batch = Batch.from_data_list(lst, batch.attrs).to(device)
    samples_batch, n = sampling_fn(model, batch)
    filename = f'{step}'
    filenmae = saveMol(samples_batch, idx=0, workdir=FLAGS.output_path, filename=filename)

def main(argv):
  torch.cuda.set_device(0)
  torch.cuda.empty_cache()
  torch.jit.set_fusion_strategy([('DYNAMIC', 3)])
  FLAGS = flags.FLAGS

  config_name = FLAGS.config
  e3_config = getattr(configs, config_name, None)
  assert not e3_config is None, f"Config {config_name} not found."
  e3_config = e3_config(FLAGS.config_spec)
      
  if FLAGS.sde_config is None:
    config_dict = e3_config.to_dict()
  else:
    config_dict = {'e3': e3_config.to_dict(), 'sde': FLAGS.sde_config.to_dict()}
  os.environ["MASTER_ADDR"] = FLAGS.master_addr
  os.environ["MASTER_PORT"] = FLAGS.master_port
  dist.init_process_group("nccl", rank=0, world_size=1)
  setSeed(FLAGS.seed)
  if FLAGS.sde_config is None:
    regression(e3_config, FLAGS)
  else:
    FLAGS.sde_config.training.batch_size = e3_config.batch_size
    diffusion(e3_config, FLAGS)


if __name__ == "__main__":
    app.run(main)