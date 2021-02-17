# FullSubNet & Fullband Baseline

## Dataset

### Deep Noise Suppression Challenge - INTERSPEECH 2020 (DNS-INTERSPEECH-2020)

The reported performance of FullSuBNet in the paper is on this dataset. It consists of 65000 speech clips (30s per clip) and 65000 noise clips (10s
per clip). You can download this dataset in [https://github.com/microsoft/DNS-Challenge.git](https://github.com/microsoft/DNS-Challenge.git).

This Git repository contains the DNS Challenge dataset (INTERSPEECH 2020) and the newer DNS Challenge dataset (ICASSP 2021). The default branch of the
Git repository is the ICASSP 2021 Dataset. You need to check out the default branch to `interspeech2020` branch.

## Usage

### Training

You can use the default training configuration:

```shell
cd FullSubNet/src/dns_interspeech_2020

# Use default config and two GPUs to train the FullSubNet model
CUDA_VISIABLE_DEVICES=0,1 python train.py -C fullsubnet/train.toml -N 2

# Use default config and one GPU to train the Fullband baseline model
CUDA_VISIABLE_DEVICES=0 python train.py -C fullband_baseline/train.toml -N 1

# Resume the experiment
CUDA_VISIABLE_DEVICES=0,1 python train.py -C fullband_baseline/train.toml -W 2 -R
```

### Inference

TODO

### Apply a Pre-Trained Model

TODO

### Logs and Visualization

Assuming that:

- the file path of the training configuration is `config/train/fullsubnet_baseline.toml`
- In the training configuration, `["meta"]["save_dir"] = "~/Experiments/FullSubNet/"`

The logs will be stored in the `~/Experiments/FullSubNet/fullsubnet_baseline` directory. This directory contains the following:

- `logs/` directory: store the Tensorboard related data, including loss curves, audio files, and spectrogram figures.
- `checkpoints/` directory: stores all checkpoints of the model, from which you can resume the training or start an inference.
- `*.toml` file: the backup of the training configuration.

In the `logs/` directory, use the following command to visualize loss curves, spectrogram figures, and audio files during the training and the
validation.

```shell
tensorboard --logdir ~/Experiments/FullSubNet/fullsubnet_baseline

# specify a port
tensorboard --logdir ~/Experiments/FullSubNet/fullsubnet_baseline --port 45454
```

### Metrics

```shell
# DNS-INTERSPEECH-2020
python tools/calculate_metrics.py \
  -R ~/Datasets/DNS-Challenge-INTERSPEECH/datasets/test_set/synthetic/no_reverb/clean \
  -E ~/Enhancement/fullsubnet_dns_interspeech_no_reverb/enhanced_1155 \
  -M SI_SDR,STOI,WB_PESQ,NB_PESQ \
  -S DNS_1
```