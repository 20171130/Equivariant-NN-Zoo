from absl import app
from absl import flags

import os
from pathlib import Path
import logging
from tqdm import tqdm

import torch

from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import CondensedDataset, DataLoader, Batch


FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "The name of the config.")
flags.DEFINE_string("config_spec", 'eval', "Config specification, the argument of get_config().")
flags.DEFINE_string("output_path", "results.hdf5", "Path to the output file to create.")
flags.DEFINE_string("name", "default", "Name of the experiment.")
flags.DEFINE_string("model_path", None, "The name of the model checkpoint.")
flags.DEFINE_list("output_keys", [], "The output keys to save.")
flags.DEFINE_integer("seed", None, "The RNG seed.")
flags.DEFINE_integer(
    "dataloader_num_workers", 4, "Number of workers per training process."
)
flags.DEFINE_boolean(
    "equivariance_test",
    False,
    "Whether to test the equivariance of the neural network.",
)

flags.DEFINE_string("verbose", "INFO", "Logging verbosity.")
flags.mark_flags_as_required(["config"])


def evaluate(argv):
    config_name = FLAGS.config
    config = getattr(configs, config_name, None)
    assert not config is None, f"Config {config_name} not found."
    config = config(FLAGS.config_spec)
    
    model = build(config.model_config)
    device = torch.device('cuda')
    model.to(device=device)
    if FLAGS.model_path:
        model_state_dict = torch.load(FLAGS.model_path, map_location=device)
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
    with torch.no_grad():
        lst = []
        for batch in tqdm(iter(dataloader)):
            batch = batch.to(device)
            out = model(batch.clone())
            dic = {key: out[key] for key in FLAGS.output_keys}
            lst.append(dic)
    result = Batch.from_data_list(lst, attrs = batch.attrs)
    result.dumpHDF5(FLAGS.output_path)

if __name__ == "__main__":
    app.run(evaluate)
