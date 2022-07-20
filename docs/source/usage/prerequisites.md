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

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# create a conda environment
conda create --name FullSubNet python=3
conda activate FullSubNet

# install conda packages
# ensure python=3.x, pytorch=1.10.x, torchaudio=0.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install tensorboard joblib matplotlib

# install pip packages
pip install Cython
pip install librosa pesq pypesq pystoi tqdm toml mir_eval torch_complex rich

# (Optional) if there are "mp3" format audio files in your dataset, you need to install ffmpeg.
conda install -c conda-forge ffmpeg
```