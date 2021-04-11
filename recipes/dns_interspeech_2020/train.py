import argparse
import os
import random
import sys
from socket import socket

import numpy as np
import toml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
import audio_zen.loss as loss
from audio_zen.utils import initialize_module


def entry(rank, world_size, config, resume, only_validation):
    torch.manual_seed(config["meta"]["seed"])  # For both CPU and GPU
    np.random.seed(config["meta"]["seed"])
    random.seed(config["meta"]["seed"])

    os.environ["MASTER_ADDR"] = "localhost"
    s = socket()
    s.bind(("", 0))
    os.environ["MASTER_PORT"] = "1111"  # A random local port

    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # The DistributedSampler will split the dataset into the several cross-process parts.
    # On the contrary, "Sampler=None, shuffle=True", each GPU will get all data in the whole dataset.
    train_dataloader = DataLoader(
        dataset=initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"]),
        **config["train_dataset"]["dataloader"],
    )

    valid_dataloader = DataLoader(
        dataset=initialize_module(config["validation_dataset"]["path"], args=config["validation_dataset"]["args"]),
        num_workers=0,
        batch_size=1
    )

    model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = getattr(loss, config["loss_function"]["name"])(**config["loss_function"]["args"])
    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    trainer = trainer_class(
        dist=dist,
        rank=rank,
        config=config,
        resume=resume,
        only_validation=only_validation,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FullSubNet")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.toml).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume the experiment from latest checkpoint.")
    parser.add_argument("-V", "--only_validation", action="store_true", help="Only run validation. It is used for debugging validation.")
    parser.add_argument("-N", "--num_gpus", type=int, default=2, help="The number of GPUs you are using for training.")
    parser.add_argument("-P", "--preloaded_model_path", type=str, help="Path of the *.pth file of a model.")
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert not args.resume, "The 'resume' conflicts with the 'preloaded_model_path'."

    configuration = toml.load(args.configuration)

    configuration["meta"]["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["meta"]["config_path"] = args.configuration
    configuration["meta"]["preloaded_model_path"] = args.preloaded_model_path

    # Expand python search path to "recipes"
    # sys.path.append(os.path.join(os.getcwd(), ".."))

    # One training job is corresponding to one group (world).
    # The world size is the number of processes for training, which is usually the number of GPUs you are using for distributed training.
    # the rank is the unique ID given to a process.
    # Find more information about DistributedDataParallel (DDP) in https://pytorch.org/tutorials/intermediate/ddp_tutorial.html.
    mp.spawn(entry,
             args=(args.num_gpus, configuration, args.resume, args.only_validation),
             nprocs=args.num_gpus,
             join=True)
