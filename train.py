import argparse
import os
import random
import sys

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader

from src.util.utils import initialize_module, merge_config


def main(config, resume):
    torch.manual_seed(config["meta"]["seed"])  # For all devices (both CPU and CUDA)
    np.random.seed(config["meta"]["seed"])
    random.seed(config["meta"]["seed"])

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

    loss_function = initialize_module(config["loss_function"]["path"])
    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FullSubNet")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json5).")
    parser.add_argument("-P", "--preloaded_model_path", type=str, help="Path of the *.Pth file of the model.")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    if args.preloaded_model_path:
        assert not args.resume, "'resume' conflicts with 'preloaded_model_path'."

    custom_config = toml.load(args.configuration)
    assert custom_config["inherit"], f"The config file should inherit from 'config/common/*.toml'"
    common_config = toml.load(custom_config["inherit"])
    del custom_config["inherit"]
    configuration = merge_config(common_config, custom_config)

    configuration["meta"]["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["meta"]["config_path"] = args.configuration
    configuration["meta"]["preloaded_model_path"] = args.preloaded_model_path

    sys.path.append(os.path.join(os.getcwd(), "src"))
    main(configuration, resume=args.resume)
