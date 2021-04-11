# Getting Started

## Dataset

### Deep Noise Suppression Challenge - INTERSPEECH 2020 (DNS-INTERSPEECH-2020)

The reported performance of the FullSuBNet in the paper is on this dataset.

It consists of 65000 speech clips (the 30s per clip) and 65000 noise clips (10s per clip). You can download this dataset
from [https://github.com/microsoft/DNS-Challenge.git](https://github.com/microsoft/DNS-Challenge.git).

This Git repository contains the DNS Challenge dataset (INTERSPEECH 2020) and the newer DNS Challenge dataset (ICASSP 2021). The default branch of the
Git repository is the ICASSP 2021 Dataset. You need to check out the default branch to the `interspeech2020` branch.

## Usage

### Training

Use the default training configuration:

```shell
cd FullSubNet/recipes/dns_interspeech_2020

# Use a default config and two GPUs to train the FullSubNet model
CUDA_VISIABLE_DEVICES=0,1 python train.py -C fullsubnet/train.toml -N 2

# Use default config and one GPU to train the Fullband baseline model
CUDA_VISIABLE_DEVICES=0 python train.py -C fullband_baseline/train.toml -N 1

# Resume the experiment using "-R" parameter
CUDA_VISIABLE_DEVICES=0,1 python train.py -C fullband_baseline/train.toml -N 2 -R
```

### Logs and Visualization

Assuming that:

- The file path of the training configuration is `fullsubnet/train.toml`
- In the training configuration, `save_dir = "~/Experiments/FullSubNet/FullSubNet"`

The logs will be stored in the `~/Experiments/FullSubNet/FullSubNet/train` directory. This directory contains the following:

- `logs/` directory: store the Tensorboard related data, including loss curves, audio files, and spectrogram figures.
- `checkpoints/` directory: stores all checkpoints of the model, from which you can resume the training or start an inference.
- `*.toml` file: the backup of the training configuration.

In the `logs/` directory, use the following command to visualize loss curves, spectrogram figures, and audio files during the training and the
validation.

```shell
tensorboard --logdir ~/Experiments/FullSubNet/FullSubNet/train

# specify a port
tensorboard --logdir ~/Experiments/FullSubNet/FullSubNet/train --port 45454
```

### Inference or Applying a Pre-trained Model

After training, you can enhance noisy speech. Take the FullSubNet as an example:

1. Checking the noisy speech directory path and the sample rate in `fullsubnet/inference.toml`.

```toml
[dataset.args]
dataset_dir_list = [
    "/path/to/your/dataset_1/",
    "/path/to/your/dataset_2/",
    "others..."
]
sr = 16000
```

2. Switching current path to `FullSubNet/recipes/dns_interspeech_2020`. Entering the following commandline:

```shell
cd FullSubNet/recipes/dns_interspeech_2020

# One GPU is used by default
python inference.py \
  -C fullsubnet/inference.toml \
  -M /path/to/your/checkpoint_dir/best_model.tar \
  -O /path/to/your/enhancement/dir
```

Or, you use a pre-trained model downloaded from the [Releases Page](https://github.com/haoxiangsnr/FullSubNet/releases):

1. As mentioned above, you need to check the noisy speech directory path and the sample rate in `fullsubnet/inference.toml`.
2. Changing the "-M" parameter to the path of the pre-trained model downloaded from the releases.

Check other details in `fullsubnet/inference.toml`.

### Metrics

Calculating metrics (SI_SDR, STOI, WB_PESQ, NB_PESQ, etc.) using the following commandline:

```shell
# Switching path
cd FullSubNet

# DNS-INTERSPEECH-2020
python tools/calculate_metrics.py \
  -R /path/to/reference/ \
  -E /path/to/enhancement/ \
  -M SI_SDR,STOI,WB_PESQ,NB_PESQ \
  -S DNS_1
```
