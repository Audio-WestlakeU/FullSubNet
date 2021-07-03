# FullSubNet

![Platform](https://img.shields.io/badge/Platform-macos%20%7C%20linux-lightgrey)
![Python version](https://img.shields.io/badge/Python-%3E%3D3.8.0-orange)
![Pytorch Version](https://img.shields.io/badge/PyTorch-%3E%3D1.7-brightgreen)
![GitHub repo size](https://img.shields.io/github/repo-size/haoxiangsnr/FullSubNet)

This Git repository for the official PyTorch implementation
of ["FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement"](https://arxiv.org/abs/2010.15508), accepted
to ICASSP 2021.

[![FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement](https://res.cloudinary.com/marcomontalbano/image/upload/v1625280913/video_to_markdown/images/youtube--XJeE-MWDlk0-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/XJeE-MWDlk0 "FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement")

<p align="center">
:bulb:[[Demo\]](https://www.haoxiangsnr.com/demo/fullsubnet/) | :page_facing_up:[[PDF\]](https://arxiv.org/abs/2010.15508) | :floppy_disk:[[Model Checkpoint\]](https://github.com/haoxiangsnr/FullSubNet/releases) | :satellite:[[Loss Curve\]](https://tensorboard.dev/experiment/63WgyAXOSbiBzHg4AdVfYw/#scalars)
</p>

<p align="center">
  <img width="460" src="docs/workflow.png" alt="workflow">
</p>

![fullsubnet_result](docs/fullsubnet-result.png)

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
    volume={},
    number={},
    pages={6633-6637},
    doi={10.1109/ICASSP39728.2021.9414177}
}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/haoxiangsnr/FullSubNet/blob/main/LICENSE)

