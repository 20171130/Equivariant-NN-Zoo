from absl import app
from absl import flags

import os
from pathlib import Path
from shutil import rmtree
import logging

import torch.distributed as dist
import torch.multiprocessing as mp

import wandb

from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import CondensedDataset


FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "The name of the config.")
flags.DEFINE_string("config_spec", None, "Config specification, the argument of get_config().")
flags.DEFINE_string("workdir", "results", "Directory to store the outputs.")
flags.DEFINE_string("name", "default", "Name of the experiment.")
flags.DEFINE_string("resume_from", None, "The name of the trainer checkpoint to resume from.")
flags.DEFINE_integer("seed", None, "The RNG seed.")
flags.DEFINE_integer(
    "dataloader_num_workers", 4, "Number of workers per training process."
)
flags.DEFINE_boolean(
    "equivariance_test",
    False,
    "Whether to test the equivariance of the neural network.",
)

flags.DEFINE_integer("world_size", 1, "Number of processes.")
flags.DEFINE_string("master_addr", "127.0.0.1", "The address to use.")
flags.DEFINE_string("master_port", "10000", "The port to use.")

flags.DEFINE_boolean("wandb", False, "If logging with wandb.")
flags.DEFINE_boolean("profiling", False, "If profiling.")
flags.DEFINE_string("wandb_project", None, "The name of the wandb project.")
flags.DEFINE_string("verbose", "INFO", "Logging verbosity.")
flags.DEFINE_integer("log_period", 100, "Number of batches.")
flags.DEFINE_integer("save_period", 1, "Number of epoches.")

flags.mark_flags_as_required(["config"])


def run(rank, config):
    FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.name)
    if rank == 0:
        if not FLAGS.resume_from and os.path.isdir(FLAGS.workdir):
            rmtree(FLAGS.workdir)
        Path(FLAGS.workdir).mkdir(parents=True, exist_ok=True)

    mp.set_start_method("fork", force=True)
    dist.init_process_group("nccl", rank=rank, world_size=FLAGS.world_size)

    try:
        if FLAGS.wandb and rank == 0:
            from e3_layers.run.trainer import TrainerWandB as Trainer

            config_dict = config.to_dict()
            run_id = wandb.util.generate_id()
            wandb.init(
                project=FLAGS.wandb_project,
                config=config_dict,
                name=f"{FLAGS.name}_{FLAGS.seed}",
                resume="allow",
                id=run_id,
                dir = FLAGS.workdir,
                settings=wandb.Settings(),
            )
        else:
            from e3_layers.run.trainer import Trainer

        if not FLAGS.resume_from:
            model = build(config.model_config)
            trainer = Trainer(model=model, **dict(config))
        else:
            trainer = Trainer.from_file(FLAGS.resume_from, **dict(config))
        data_config = config.data_config
        dataset = CondensedDataset(**data_config)
        trainer.set_dataset(dataset, validation_dataset=None)
        logging.info("Successfully built the network...")

        if rank == 0:
            trainer.save()
        trainer.train()
    except BaseException as e:
        dist.destroy_process_group()
        raise e


def launch_mp(argv):
    config_name = FLAGS.config
    config = getattr(configs, config_name, None)
    assert not config is None, f"Config {config_name} not found."
    config = config(FLAGS.config_spec)
    world_size = FLAGS.world_size
    os.environ["MASTER_ADDR"] = FLAGS.master_addr
    os.environ["MASTER_PORT"] = FLAGS.master_port
    if world_size == 1:
        run(0, config)
    else:
        mp.spawn(run, args=(config,), nprocs=config.world_size, join=True)


if __name__ == "__main__":
    app.run(launch_mp)
