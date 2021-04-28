import time
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import toml
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_zen.acoustics.feature import stft, istft, mc_stft
from audio_zen.utils import initialize_module, prepare_device, prepare_empty_dir


class BaseInferencer:
    def __init__(self, config, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        root_dir = Path(output_dir).expanduser().absolute()
        self.device = prepare_device(torch.cuda.device_count())

        print("Loading inference dataset...")
        self.dataloader = self._load_dataloader(config["dataset"])
        print("Loading model...")
        self.model, epoch = self._load_model(config["model"], checkpoint_path, self.device)
        self.inference_config = config["inferencer"]

        self.enhanced_dir = root_dir / f"enhanced_{str(epoch).zfill(4)}"
        prepare_empty_dir([self.enhanced_dir])

        # Acoustics
        self.acoustic_config = config["acoustics"]

        # Supported STFT
        self.n_fft = self.acoustic_config["n_fft"]
        self.hop_length = self.acoustic_config["hop_length"]
        self.win_length = self.acoustic_config["win_length"]
        self.sr = self.acoustic_config["sr"]

        # See utils_backup.py
        self.torch_stft = partial(stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.torch_istft = partial(istft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.torch_mc_stft = partial(mc_stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.librosa_stft = partial(librosa.stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.librosa_istft = partial(librosa.istft, hop_length=self.hop_length, win_length=self.win_length)

        print("Configurations are as follows: ")
        print(toml.dumps(config))
        with open((root_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
            toml.dump(config, handle)

    @staticmethod
    def _load_dataloader(dataset_config):
        dataset = initialize_module(dataset_config["path"], args=dataset_config["args"], initialize=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @staticmethod
    def _unfold(input, pad_mode, n_neighbor):
        """
        沿着频率轴，将语谱图划分为多个 overlap 的子频带

        Args:
            input: [B, C, F, T]

        Returns:
            [B, N, C, F, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}, which should be 4."
        batch_size, n_channels, n_freqs, n_frames = input.size()
        output = input.reshape(batch_size * n_channels, 1, n_freqs, n_frames)
        sub_band_n_freqs = n_neighbor * 2 + 1

        output = functional.pad(output, [0, 0, n_neighbor, n_neighbor], mode=pad_mode)
        output = functional.unfold(output, (sub_band_n_freqs, n_frames))
        assert output.shape[-1] == n_freqs, f"n_freqs != N (sub_band), {n_freqs} != {output.shape[-1]}"

        # 拆分 unfold 中间的维度
        output = output.reshape(batch_size, n_channels, sub_band_n_freqs, n_frames, n_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # permute 本质上与  reshape 可是不同的 ...，得到的维度相同，但 entity 不同啊
        return output

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):
        model = initialize_module(model_config["path"], args=model_config["args"], initialize=True)
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(f"当前正在处理 tar 格式的模型断点，其 epoch 为：{epoch}.")

        model.load_state_dict(model_static_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"]

    @torch.no_grad()
    def multi_channel_mag_to_mag(self, noisy, inference_args=None):
        """
        模型的输入为带噪语音的 **幅度谱**，输出同样为 **幅度谱**
        """
        mixture_stft_coefficients = self.torch_mc_stft(noisy)
        mixture_mag = (mixture_stft_coefficients.real ** 2 + mixture_stft_coefficients.imag ** 2) ** 0.5

        enhanced_mag = self.model(mixture_mag)

        # Phase of the reference channel
        reference_channel_stft_coefficients = mixture_stft_coefficients[:, 0, ...]
        noisy_phase = torch.atan2(reference_channel_stft_coefficients.imag, reference_channel_stft_coefficients.real)
        complex_tensor = torch.stack([(enhanced_mag * torch.cos(noisy_phase)), (enhanced_mag * torch.sin(noisy_phase))], dim=-1)
        enhanced = self.torch_istft(complex_tensor, length=noisy.shape[-1])

        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def __call__(self):
        inference_type = self.inference_config["type"]
        assert inference_type in dir(self), f"Not implemented Inferencer type: {inference_type}"

        inference_args = self.inference_config["args"]

        for noisy, name, noisy_file_path, dataset_dir in tqdm(self.dataloader, desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            enhanced = getattr(self, inference_type)(noisy.to(self.device), inference_args)

            if abs(enhanced).any() > 1:
                print(f"Warning: enhanced is not in the range [-1, 1], {name}")

            amp = np.iinfo(np.int16).max
            enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))

            # clnsp102_traffic_248091_3_snr0_tl-21_fileid_268 => clean_fileid_0
            # name = "clean_" + "_".join(name.split("_")[-2:])
            if dataset_dir is not None:
                noisy_file_path = noisy_file_path[0]
                dataset_dir = dataset_dir[0]
                out_path = Path(str(noisy_file_path).replace(str(dataset_dir), str(self.enhanced_dir)))
                prepare_empty_dir([out_path.parents[0]])
                sf.write(out_path, enhanced, samplerate=self.acoustic_config["sr"])
            else:
                sf.write(self.enhanced_dir / f"{name}.wav", enhanced, samplerate=self.acoustic_config["sr"])