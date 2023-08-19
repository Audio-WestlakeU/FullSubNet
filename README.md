<div align="center">
    <h1>
        FullSubNet
    </h1>
    <p>
    Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement
    </p>
    <a href="https://github.com/haoxiangsnr/FullSubNet/"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/haoxiangsnr/FullSubNet/"><img src="https://img.shields.io/github/stars/haoxiangsnr/FullSubNet?color=yellow&amp;label=FullSubNet&amp;logo=github" alt="Generic badge"></a>
    <a href='https://fullsubnet.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/fullsubnet/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/haoxiangsnr/FullSubNet/"><img src="https://img.shields.io/badge/Python-3.10-orange" alt="version"></a>
    <a href="https://github.com/haoxiangsnr/FullSubNet/"><img src="https://img.shields.io/badge/PyTorch-1.12-brightgreen" alt="python"></a>
    <a href="https://github.com/haoxiangsnr/FullSubNet/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="mit"></a>
</div>

## Guides

The documentation is hosted on [Read the Docs](https://fullsubnet.readthedocs.io/). Check the documentation for **how to train and test models**.

- Improved FullSubNet: Further reduces computational costs and enables high sampling rate data processing, e.g., 48 KHz and 24 KHz.
  - ❇️ [Model Architecture](https://github.com/Audio-WestlakeU/FullSubNet/blob/main/recipes/dns_interspeech_2020/improved_fullsubnet/model.py)
- 📰 [FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement, ICASSP 2021](https://arxiv.org/abs/2010.15508)
  - 📸 [Demo (Audio Clips)](https://www.haoxiangsnr.com/publications/3)
  - 🎏 [Model Checkpoints](https://github.com/haoxiangsnr/FullSubNet/releases)
  - ❇️ [Model Architecture](https://github.com/haoxiangsnr/FullSubNet/blob/fast_fullsubnet/recipes/dns_interspeech_2020/fullsubnet/model.py)
- 📰 [Fast FullSubNet: Accelerate Full-band and Sub-band Fusion Model for Single-channel Speech Enhancement](https://arxiv.org/abs/2212.09019)
  - ❇️ [Model Architecture](https://github.com/haoxiangsnr/FullSubNet/blob/fast_fullsubnet/recipes/dns_interspeech_2020/fast_fullsubnet/model.py)
  - 📸 [Demo (Audio Clips)](https://www.haoxiangsnr.com/publications/3)
- cIRM-based Fullband baseline model (described in the original FullSubNet paper)
  - ❇️ [Model Architecture](https://github.com/haoxiangsnr/FullSubNet/blob/fast_fullsubnet/recipes/dns_interspeech_2020/fullband_baseline/model.py)


## Citation

If you use this code for your research, please consider citeing:

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

This repository Under the [MIT license](LICENSE).
