# FullSubNet

![Platform](https://img.shields.io/badge/Platform-linux-lightgrey)
![Python version](https://img.shields.io/badge/Python-%3E%3D3.8.0-orange)
![Pytorch Version](https://img.shields.io/badge/PyTorch-%3E%3D1.10-brightgreen)
![GitHub repo size](https://img.shields.io/github/repo-size/haoxiangsnr/FullSubNet)

This Git repository for the official PyTorch implementation
of ["FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement"](https://arxiv.org/abs/2010.15508), accepted
to ICASSP 2021.

:bulb:[[Demo\]](https://www.haoxiangsnr.com/demo/fullsubnet/) | :page_facing_up:[[PDF\]](https://arxiv.org/abs/2010.15508) | :floppy_disk:[[Model Checkpoint\]](https://github.com/haoxiangsnr/FullSubNet/releases)

## Introduction

[![Click it to show a video](https://i.imgur.com/s3mq7NNl.png)](https://youtu.be/XJeE-MWDlk0 "FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement")

## Key Features

You can use all of these things:

- Available models
  - [x] Fullband Baseline
  - [x] FullSubNet
  - [ ] FullSubNet (lightweight)
  - [ ] Delayed Sub-Band LSTM

- Available datasets
  - [x] Deep Noise Suppression Challenge - INTERSPEECH 2020
  - [ ] Demand + CSTR VCTK Corpus

## Documentation

- [Prerequisites](docs/prerequisites.md)
- [Getting Started](docs/getting_started.md)

## Citation

If you use this code for your research, please consider citing:

```text
@INPROCEEDINGS{hao2020fullsubnet,
    author={Hao, Xiang and Su, Xiangdong and Horaud, Radu and Li, Xiaofei},
    booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Fullsubnet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement}, 
    year={2021},
    pages={6633-6637},
    doi={10.1109/ICASSP39728.2021.9414177}
}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/haoxiangsnr/FullSubNet/blob/main/LICENSE)

