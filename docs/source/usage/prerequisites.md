# Prerequisites

- Linux-based system
- Anaconda or Miniconda
- NVIDIA GPU + CUDA CuDNN (CPU is **not** be supported)

In order to run experiments, you need to use a Linux-based operating system (not supported on Windows platforms). To train a model using GPUs, you need to install CUDA (10.2+) in your operating system. In addition, I recommend that you install Anaconda or Miniconda, which are used to create virtual environments and install dependencies.
The advantage of using conda instead of pip is that conda will ensure that you have the appropriate version of the CuDNN and other packages.

## Clone

Firstly, clone this repository:

```shell
git clone https://github.com/haoxiangsnr/FullSubNet
cd FullSubNet
```

## Environment && Installation

After install Anaconda or Miniconda, you are able to install conda and pip packages:

### Create a Conda Environment

```shell
# (Optional) if you already have an environment with the same name, you may remove it first
conda env remove --name FullSubNet

# create a new environment
conda create --name FullSubNet python=3.10
conda activate FullSubNet
```

### Install conda packages

As recommended by the Anaconda official, you may install conda package first before installing other pip packages.
Generally speaking, the CUDA version is not deterministic. You may choose the CUDA version you want.
This project uses the latest `pytorch=1.12.0` as the default PyTorch version. Other PyTorch versions may work, but I haven't tested them.

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install tensorboard joblib matplotlib -c conda-forge
```

### Install Pypi packages

```shell
# you may need to install Cython firstly
pip install Cython

pip install librosa tbb tensorboard joblib matplotlib pesq pystoi tqdm toml torch_complex rich

# install pypesq directly from the GitHub repository
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

# (Optional) if there are "mp3" format audio files in your dataset, you need to install ffmpeg
conda install -c conda-forge ffmpeg
```