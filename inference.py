import argparse
import os
import sys

import toml

from src.util.utils import initialize_module, merge_config


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Configuration file.")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="The path of your model checkpoint.")
    parser.add_argument("-O", "--output_dir", type=str, required=True, help="The path to save the enhanced speech.")
    args = parser.parse_args()

    custom_config = toml.load(args.configuration)
    assert custom_config["inherit"], f"The config file should inherit from 'config/common/*.toml'"
    common_config = toml.load(custom_config["inherit"])
    del custom_config["inherit"]
    configuration = merge_config(common_config, custom_config)

    checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir

    sys.path.append(os.path.join(os.getcwd(), "src"))
    main(configuration, checkpoint_path, output_dir)
