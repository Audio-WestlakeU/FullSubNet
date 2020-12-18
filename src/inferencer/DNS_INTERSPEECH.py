import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from common.inferencer import BaseInferencer
from util.acoustic_utils import mag_phase


def cumulative_norm(input):
    eps = 1e-10
    device = input.device
    data_type = input.dtype
    n_dim = input.ndim

    assert n_dim in (3, 4)

    if n_dim == 3:
        n_channels = 1
        batch_size, n_freqs, n_frames = input.size()
    else:
        batch_size, n_channels, n_freqs, n_frames = input.size()
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

    step_sum = torch.sum(input, dim=1)  # [B, T]
    step_pow_sum = torch.sum(torch.square(input), dim=1)

    cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
    cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

    entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
    entry_count = entry_count.reshape(1, n_frames)  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cum_mean = cumulative_sum / entry_count  # B, T
    cum_var = (cumulative_pow_sum - 2 * cum_mean * cumulative_sum) / entry_count + cum_mean.pow(2)  # B, T
    cum_std = (cum_var + eps).sqrt()  # B, T

    cum_mean = cum_mean.reshape(batch_size * n_channels, 1, n_frames)
    cum_std = cum_std.reshape(batch_size * n_channels, 1, n_frames)

    x = (input - cum_mean) / cum_std

    if n_dim == 4:
        x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

    return x


class Inferencer(BaseInferencer):
    def __init__(self, config, checkpoint_path, output_dir):
        super().__init__(config, checkpoint_path, output_dir)

    @torch.no_grad()
    def mag(self, noisy, inference_args):
        noisy_complex = self.stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)  # [B, F, T] => [B, 1, F, T]

        enhanced_mag = self.model(noisy_mag.unsqueeze(1)).squeeze(1)

        enhanced = self.istft((enhanced_mag, noisy_phase), length=noisy.size(-1), use_mag_phase=True)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def scaled_mask(self, noisy, inference_args):
        noisy_complex = self.stft(noisy)
        noisy_mag, noisy_phase = mag_phase(noisy_complex)

        # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
        noisy_mag = noisy_mag.unsqueeze(1)
        scaled_mask = self.model(noisy_mag)
        scaled_mask = scaled_mask.permute(0, 2, 3, 1)

        enhanced_complex = noisy_complex * scaled_mask
        enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def sub_band_crm_mask(self, noisy, inference_args):
        pad_mode = inference_args["pad_mode"]
        n_neighbor = inference_args["n_neighbor"]

        noisy = noisy.cpu().numpy().reshape(-1)
        noisy_D = self.librosa_stft(noisy)

        noisy_real = torch.tensor(noisy_D.real, device=self.device)
        noisy_imag = torch.tensor(noisy_D.imag, device=self.device)
        noisy_mag = torch.sqrt(torch.square(noisy_real) + torch.square(noisy_imag))  # [F, T]
        n_freqs, n_frames = noisy_mag.size()

        noisy_mag = noisy_mag.reshape(1, 1, n_freqs, n_frames)
        noisy_mag_padded = self._unfold(noisy_mag, pad_mode, n_neighbor)  # [B, N, C, F_s, T] <=> [1, 257, 1, 31, T]
        noisy_mag_padded = noisy_mag_padded.squeeze(0).squeeze(1)  # [257, 31, 200] <=> [B, F_s, T]

        pred_crm = self.model(noisy_mag_padded).detach()  # [B, 2, T] <=> [F, 2, T]
        pred_crm = pred_crm.permute(0, 2, 1).contiguous()  # [B, T, 2]

        lim = 9.99
        pred_crm = lim * (pred_crm >= lim) - lim * (pred_crm <= -lim) + pred_crm * (torch.abs(pred_crm) < lim)
        pred_crm = -10 * torch.log((10 - pred_crm) / (10 + pred_crm))

        enhanced_real = pred_crm[:, :, 0] * noisy_real - pred_crm[:, :, 1] * noisy_imag
        enhanced_imag = pred_crm[:, :, 1] * noisy_real + pred_crm[:, :, 0] * noisy_imag

        enhanced_real = enhanced_real.cpu().numpy()
        enhanced_imag = enhanced_imag.cpu().numpy()
        enhanced = self.librosa_istft(enhanced_real + 1j * enhanced_imag, length=len(noisy))
        return enhanced

    @torch.no_grad()
    def full_band_crm_mask(self, noisy, inference_args):
        noisy_complex = self.stft(noisy)

        noisy_mag, _ = mag_phase(noisy_complex)

        noisy_mag = noisy_mag.unsqueeze(1)
        pred_crm = self.model(noisy_mag)
        pred_crm = pred_crm.permute(0, 2, 3, 1)

        lim = 9.9
        pred_crm = lim * (pred_crm >= lim) - lim * (pred_crm <= -lim) + pred_crm * (torch.abs(pred_crm) < lim)
        pred_crm = -10 * torch.log((10 - pred_crm) / (10 + pred_crm))

        enhanced_real = pred_crm[..., 0] * noisy_complex[..., 0] - pred_crm[..., 1] * noisy_complex[..., 1]
        enhanced_imag = pred_crm[..., 1] * noisy_complex[..., 0] + pred_crm[..., 0] * noisy_complex[..., 1]
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

        enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)

        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        return enhanced

    @torch.no_grad()
    def __call__(self):
        inference_type = self.inference_config["type"]
        assert inference_type in dir(self), f"Not implemented Inferencer type: {inference_type}"

        inference_args = self.inference_config["args"]

        for noisy, name in tqdm(self.dataloader, desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            enhanced = getattr(self, inference_type)(noisy.to(self.device), inference_args)

            if abs(enhanced).any() > 1:
                print(f"Warning: enhanced is not in the range [-1, 1], {name}")

            amp = np.iinfo(np.int16).max
            enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))

            # clnsp102_traffic_248091_3_snr0_tl-21_fileid_268 => clean_fileid_0
            # name = "clean_" + "_".join(name.split("_")[-2:])
            sf.write(self.enhanced_dir / f"{name}.wav", enhanced, samplerate=self.acoustic_config["sr"])


if __name__ == '__main__':
    a = torch.rand(10, 2, 161, 200)
    print(cumulative_norm(a).shape)
