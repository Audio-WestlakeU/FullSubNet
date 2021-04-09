# Prerequisites

- Linux or macOS
- Anaconda or Miniconda
- NVIDIA GPU + CUDA CuDNN (CPU is **not** be supported)

## Clone

Clone the repository:

```shell
git clone https://github.com/haoxiangsnr/FullSubNet
cd FullSubNet
```

## Environment && Installation

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# Create conda environment
conda create --name FullSubNet python=3.8
conda activate FullSubNet

# Install conda packages
# Check python=3.8, cudatoolkit=10.2, pytorch=1.7.1, torchaudio=0.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tensorboard joblib matplotlib

# Install pip packages
# Check librosa=0.8
pip install Cython
pip install librosa pesq pypesq pystoi tqdm toml colorful mir_eval torch_complex

# (Optional) If you want to load "mp3" format audio in your dataset
conda install -c conda-forge ffmpeg
```