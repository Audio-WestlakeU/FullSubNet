import os

import librosa
import numpy as np
import torch


def get_complex_ideal_ratio_mask(noisy_complex_tensor, clean_complex_tensor):
    noisy_real = noisy_complex_tensor[..., 0]
    noisy_imag = noisy_complex_tensor[..., 1]
    clean_real = clean_complex_tensor[..., 0]
    clean_imag = clean_complex_tensor[..., 1]

    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + 1e-10
    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator
    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)
    complex_ratio_mask = compression_using_hyperbolic_tangent(complex_ratio_mask, K=10, C=0.1)

    return complex_ratio_mask


def compression_using_hyperbolic_tangent(mask, K=10, C=0.1):
    """
        (-inf, +inf) => [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def transform_pesq_range(pesq_score):
    """
    transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
    """
    return (pesq_score + 0.5) / 5


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


def stft(y, n_fft, hop_length, win_length, device):
    assert y.dim() == 2

    window = torch.hann_window(n_fft).to(device)
    return torch.stft(y, n_fft, hop_length, win_length, window=window, return_complex=False)


def istft(complex_tensor, n_fft, hop_length, win_length, device, length=None, use_mag_phase=False):
    window = torch.hann_window(n_fft).to(device)

    if use_mag_phase:
        assert isinstance(complex_tensor, tuple) or isinstance(complex_tensor, list)
        mag, phase = complex_tensor
        complex_tensor = torch.stack([(mag * torch.cos(phase)), (mag * torch.sin(phase))], dim=-1)

    return torch.istft(complex_tensor, n_fft, hop_length, win_length, window, length=length)


def mag_phase(complex_tensor):
    mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0)
    phase = torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])
    return mag, phase


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def load_wav(file, sr=16000):
    if len(file) == 2:
        return file[-1]
    else:
        return librosa.load(os.path.abspath(os.path.expanduser(file)), mono=False, sr=sr)[0]


def aligned_subsample(data_a, data_b, sub_sample_length):
    """
    Start from a random position and take a fixed-length segment from two speech samples

    Notes
        Only support one-dimensional speech signal (T,) and two-dimensional spectrogram signal (F, T)
    """
    assert data_a.shape == data_b.shape, "Inconsistent dataset size."

    dim = np.ndim(data_a)
    assert dim == 1 or dim == 2, "Only support 1D or 2D."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        if dim == 1:
            return data_a[start:end], data_b[start:end]
        else:
            return data_a[:, start:end], data_b[:, start:end]
    elif data_a.shape[-1] == sub_sample_length:
        return data_a, data_b
    else:
        length = data_a.shape[-1]
        if dim == 1:
            return (
                np.append(data_a, np.zeros(sub_sample_length - length, dtype=np.float32)),
                np.append(data_b, np.zeros(sub_sample_length - length, dtype=np.float32))
            )
        else:
            return (
                np.append(data_a, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1),
                np.append(data_b, np.zeros(shape=(data_a.shape[0], sub_sample_length - length), dtype=np.float32), axis=-1)
            )


def subsample(data, sub_sample_length):
    """
    从随机位置开始采样出指定长度的数据

    Notes
        仅支持 1D 数据
    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        data = data[start:end]
        assert len(data) == sub_sample_length
        return data
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        return data
    else:
        return data


def overlap_cat(chunk_list, dim=-1):
    """
    按照 50% 的 overlap 沿着最后一个维度对 chunk_list 进行拼接

    Args:
        dim: 需要拼接的维度
        chunk_list(list): [[B, T], [B, T], ...]

    Returns:
        overlap 拼接后
    """
    overlap_output = []
    for i, chunk in enumerate(chunk_list):
        first_half, last_half = torch.split(chunk, chunk.size(-1) // 2, dim=dim)
        if i == 0:
            overlap_output += [first_half, last_half]
        else:
            overlap_output[-1] = (overlap_output[-1] + first_half) / 2
            overlap_output.append(last_half)

    overlap_output = torch.cat(overlap_output, dim=dim)
    return overlap_output


def activity_detector(audio, fs=16000, activity_threshold=0.13, target_level=-25, eps=1e-6):
    """
    Return the percentage of the time the audio signal is above an energy threshold

    Args:
        audio:
        fs:
        activity_threshold:
        target_level:
        eps:

    Returns:

    """
    audio, _, _ = tailor_dB_FS(audio, target_level)
    window_size = 50  # ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + eps)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > activity_threshold:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def drop_sub_band(input, num_sub_batches=3):
    """
    To reduce the computational complexity of the sub_band sub model in the FullSubNet model.

    Args:
        input: [B, C, F, T]
        num_sub_batches:

    Notes:
        'batch_size' of the input should be divisible by the value of 'num_sub_batch'.
        If not, the frequencies corresponding to the last sub batch will not be well-trained.

    Returns:
        [B, C, F // num_sub_batches, T]
    """
    if num_sub_batches < 2:
        return input

    batch_size, _, n_freqs, _ = input.shape
    sub_batch_size = batch_size // num_sub_batches
    reminder = n_freqs % num_sub_batches

    output = []
    for idx in range(num_sub_batches):
        batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=input.device)
        freq_indices = torch.arange(
            idx + (reminder // 2),
            n_freqs - (reminder - reminder // 2),
            step=num_sub_batches,
            device=input.device
        )

        selected_sub_batch = torch.index_select(input, dim=0, index=batch_indices)
        selected_freqs = torch.index_select(selected_sub_batch, dim=2, index=freq_indices)
        output.append(selected_freqs)

    return torch.cat(output, dim=0)


if __name__ == '__main__':
    ipt = torch.rand(70, 1, 257, 200)
    print(drop_sub_band(ipt, 1).shape)
