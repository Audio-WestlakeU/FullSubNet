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

from audio_zen.acoustics.feature import istft, stft
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
        self.noisy_dir = root_dir / f"noisy"

        # self.enhanced_dir = root_dir
        prepare_empty_dir([self.noisy_dir, self.enhanced_dir])

        # Acoustics
        self.acoustic_config = config["acoustics"]

        # Supported STFT
        self.n_fft = self.acoustic_config["n_fft"]
        self.hop_length = self.acoustic_config["hop_length"]
        self.win_length = self.acoustic_config["win_length"]
        self.sr = self.acoustic_config["sr"]

        # See utils_backup.py
        self.torch_stft = partial(
            stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        self.torch_istft = partial(
            istft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length
        )
        self.librosa_stft = partial(
            librosa.stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.librosa_istft = partial(
            librosa.istft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        print("Configurations are as follows: ")
        print(toml.dumps(config))
        # TODO No dump
        # with open((root_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
        #     toml.dump(config, handle)

        self.config = config

    @staticmethod
    def _load_dataloader(dataset_config):
        dataset = initialize_module(
            dataset_config["path"], args=dataset_config["args"], initialize=True
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @staticmethod
    def _unfold(input, pad_mode, n_neighbor):
        """
        Along the frequency axis, to divide the spectrogram into multiple overlapped sub band.

        Args:
            input: [B, C, F, T]

        Returns:
            [B, N, C, F, T], F is the number of frequency of the sub band unit, e.g., [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}, which should be 4."
        batch_size, n_channels, n_freqs, n_frames = input.size()
        output = input.reshape(batch_size * n_channels, 1, n_freqs, n_frames)
        sub_band_n_freqs = n_neighbor * 2 + 1

        output = functional.pad(output, [0, 0, n_neighbor, n_neighbor], mode=pad_mode)
        output = functional.unfold(output, (sub_band_n_freqs, n_frames))
        assert (
            output.shape[-1] == n_freqs
        ), f"n_freqs != N (sub_band), {n_freqs} != {output.shape[-1]}"

        # Split the middle dimensions of the unfolded features
        output = output.reshape(batch_size, n_channels, sub_band_n_freqs, n_frames, n_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()
        return output

    @staticmethod
    def _unfold_along_time(input, context_size):
        """
        Along the time axis, split overlapped chunks from spectrogram.

        Args:
            input: [B, C, F, T]
            context_size:

        Returns:
            [B, N, C, F_s, T], e.g. [2, 161, 1, 19, 200]
        """
        assert (
            input.dim() == 4
        ), f"The dims of input is {input.dim()}. It should be four-dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        # (i - N,..., i - 1, i)
        chunk_size = context_size + 1

        # [B, C, F, T] => [B * C * F, T] => [B * C * F, 1, 1, T]
        input = input.reshape(batch_size * num_channels * num_freqs, num_frames)
        input = input.unsqueeze(1).unsqueeze(1)

        # [B * C * F, chunk_size, num_chunks]
        output = functional.unfold(input, (1, chunk_size))

        # Split the dim of the unfolded feature
        # [B, num_chunks, C, F, chunk_size]
        output = output.reshape(batch_size, num_channels, num_freqs, chunk_size, -1)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):
        model = initialize_module(
            model_config["path"], args=model_config["args"], initialize=True
        )
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(f"Loading model checkpoint (epoch == {epoch})...")

        model_static_dict = {
            key.replace("module.", ""): value for key, value in model_static_dict.items()
        }

        model.load_state_dict(model_static_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"]

    @torch.no_grad()
    def __call__(self):
        inference_type = self.inference_config["type"]
        assert inference_type in dir(
            self
        ), f"Not implemented Inferencer type: {inference_type}"

        inference_args = self.inference_config["args"]

        for noisy, name in tqdm(self.dataloader, desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            enhanced = getattr(self, inference_type)(noisy.to(self.device), inference_args)

            if abs(enhanced).any() > 1:
                print(f"Warning: enhanced is not in the range [-1, 1], {name}")

            amp = np.iinfo(np.int16).max
            enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))
            sf.write(
                self.enhanced_dir / f"{name}.wav",
                enhanced,
                samplerate=self.acoustic_config["sr"],
            )

            noisy = noisy.detach().squeeze(0).numpy()
            if np.ndim(noisy) > 1:
                noisy = noisy[0, :]  # first channel
            noisy = noisy[: enhanced.shape[-1]]
            sf.write(
                self.noisy_dir / f"{name}.wav", noisy, samplerate=self.acoustic_config["sr"]
            )


if __name__ == "__main__":
    ipt = torch.rand(10, 1, 257, 100)
    opt = BaseInferencer._unfold_along_time(ipt, 30)
    print(opt.shape)
