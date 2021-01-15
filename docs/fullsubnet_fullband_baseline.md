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
# Use default config and two GPUs to train the FullSubNet model
CUDA_VISABLE_DEVICES=0,1 python train.py -C config/train/fullsubnet_baseline.toml -W 2

# Use default config and three GPUs to train the Fullband baseline model
CUDA_VISABLE_DEVICES=0,1 python train.py -C config/train/fullband_baseline.toml -W 3

# Resume the experiment
CUDA_VISABLE_DEVICES=0,1 python train.py -C config/train/fullsubnet_baseline.toml -W 2 -R
```

### Inference

When you finish the training, you can enhance the noisy speech, e.g.:

```shell
# Inference for the FullSubNet model using the best checkpoint in all training epochs.
python inference.py \
  -C config/inference/fullsubnet.toml \
  -M ~/Experiments/FullSubNet/fullsubnet_baseline/checkpoints/best_model.tar \
  -O ~/Enhancement/fullsubnet_dns_interspeech_no_reverb
  
# Inference for the Fullband Baseline model
python inference.py \
  -C config/inference/fullband.toml \
  -M ~/Experiments/FullSubNet/fullband_baseline/checkpoints/best_model.tar \
  -O ~/Enhancement/fullband_dns_interspeech_no_reverb
```

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
python scripts/calculate_metrics.py \
  -R ~/Datasets/DNS-Challenge-INTERSPEECH/datasets/test_set/synthetic/no_reverb/clean \
  -E ~/Enhancement/fullsubnet_dns_interspeech_no_reverb/enhanced_1155 \
  -M SI_SDR,STOI,WB_PESQ,NB_PESQ \
  -S DNS_1
```