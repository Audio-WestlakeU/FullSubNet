import os
from functools import partial

import librosa
import numpy as np
import torch
import torch.nn as nn


def stft(y, n_fft, hop_length, win_length):
    """
    Wrapper of the official torch.stft for single-channel and multi-channel

    Args:
        y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size

    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]

    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)

    complex_stft = torch.stft(y, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=y.device),
                              return_complex=True)
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

    mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
    real, imag = complex_stft.real, complex_stft.imag
    return mag, phase, real, imag


def istft(features, n_fft, hop_length, win_length, length=None, input_type="complex"):
    """
    Wrapper of the official torch.istft

    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")

    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
        assert isinstance(features, tuple) or isinstance(features, list)
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        assert isinstance(features, torch.ComplexType)
    elif input_type == "mag_phase":
        # the feature is (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError("Only 'real_imag', 'complex', and 'mag_phase' are supported")

    return torch.istft(features, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=features.device),
                       length=length)


def mag_phase(complex_tensor):
    mag, phase = torch.abs(complex_tensor), torch.angle(complex_tensor)
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

        Only support subsample in the last axis.
    """
    assert data_a.shape[-1] == data_b.shape[-1], "Inconsistent dataset size."

    if data_a.shape[-1] > sub_sample_length:
        length = data_a.shape[-1]
        start = np.random.randint(length - sub_sample_length + 1)
        end = start + sub_sample_length
        # data_a = data_a[..., start: end]
        return data_a[..., start:end], data_b[..., start:end]
    elif data_a.shape[-1] < sub_sample_length:
        length = data_a.shape[-1]
        pad_size = sub_sample_length - length
        pad_width = [(0, 0)] * (data_a.ndim - 1) + [(0, pad_size)]
        data_a = np.pad(data_a, pad_width=pad_width, mode="constant", constant_values=0)
        data_b = np.pad(data_b, pad_width=pad_width, mode="constant", constant_values=0)
        return data_a, data_b
    else:
        return data_a, data_b


def subsample(data, sub_sample_length, start_position: int = -1, return_start_position=False):
    """
    Randomly select fixed-length data from 

    Args:
        data: **one-dimensional data**
        sub_sample_length: how long
        start_position: If start index smaller than 0, randomly generate one index

    """
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end = start_position + sub_sample_length
        data = data[start_position:end]
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
    else:
        pass

    assert len(data) == sub_sample_length

    if return_start_position:
        return data, start_position
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


def batch_shuffle_frequency(tensor, indices=None):
    """

    Randomly shuffle frequency of a spectrogram and return shuffle indices.

    Args:
        tensor: input tensor with batch dim
        indices:

    Examples:
        input =
            tensor([[[[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.],
                      [4., 4., 4.]]],
                    [[[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.],
                      [4., 4., 4.]]]])

        output =
            tensor([[[[3., 3., 3.],
                      [4., 4., 4.],
                      [2., 2., 2.],
                      [1., 1., 1.]]],
                    [[[3., 3., 3.],
                      [2., 2., 2.],
                      [1., 1., 1.],
                      [4., 4., 4.]]]])

    Shapes:
        tensor: [B, C, F, T]
        out: [B, C, F T]
        indices: [B, C, F, T]

    Returns:
        out: after frequency shuffle
        indices: shuffle matrix
    """
    assert tensor.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = tensor.shape

    if not torch.is_tensor(indices):
        indices = torch.stack([torch.randperm(num_freqs, device=tensor.device) for _ in range(batch_size)], dim=0)
        indices = indices[:, None, :, None].repeat(1, num_channels, 1, num_frames)

    out = torch.gather(tensor, dim=2, index=indices)
    return out, indices


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        # STFT setting
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_freqs = n_fft // 2 + 1
        self.stft = partial(torch.stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.istft = partial(torch.istft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

        # IPD setting
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += self.num_freqs
            self.lps_layer_norm = ChannelWiseLayerNorm(self.num_freqs)

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_freqs * self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_freqs * self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  of shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, magnitude, phase, real, imag):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_freqs, num_frames = magnitude.shape

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel,
                            ...] ** 2 + self.eps)  # [B, F, K], the 4-th channel, which is counted from right to left.
            lps = self.lps_layer_norm(lps)
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)  # [B, I, F, K]
            cos_ipd = cos_ipd.view(batch_size, -1, num_frames)  # [B, I * F, K]
            sin_ipd = sin_ipd.view(batch_size, -1, num_frames)
            directional_feature.append(cos_ipd)
            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        return directional_feature


class ChannelDirectionalFeatureComputer(nn.Module):
    def __init__(
            self,
            n_fft,
            win_length,
            hop_length,
            input_features,
            mic_pairs,
            lps_channel,
            use_cos_IPD=True,
            use_sin_IPD=False,
            eps=1e-8
    ):
        super().__init__()
        self.eps = eps
        self.input_features = input_features

        # STFT setting
        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1

        # IPD setting
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD

        self.lps_channel = lps_channel

        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += 1

        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  pf shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)  # [B * M, T]
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape  # [B * M, F, K]

        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)

        directional_feature = []
        if "LPS" in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel,
                            ...] ** 2 + self.eps)  # [B, F, K], the 4-th channel, which is counted from right to left.
            lps = lps[:, None, ...]
            directional_feature.append(lps)

        if "IPD" in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)  # [B, I, F, K]
            directional_feature.append(cos_ipd)

            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)

        directional_feature = torch.cat(directional_feature, dim=1)

        # [B, C + I, F, T], [B, C, F, T], [B, C, F, T]
        return directional_feature, magnitude, phase, real, imag


def hz_to_bark(hz):
    return 26.81 / (1 + 1960. / hz) - 0.53


def bark_to_hz(bark):
    return 1960. / (26.81 / (0.53 + bark) - 1)


def bark_filter_bank(num_filters, n_fft, sr, low_freq, high_freq):
    high_freq = high_freq or sr / 2
    assert high_freq <= sr / 2, "highfreq is greater than samplerate/2"

    low_bark = hz_to_bark(low_freq)
    high_bark = hz_to_bark(high_freq)
    barkpoints = np.linspace(low_bark, high_bark, num_filters + 2)
    bin = np.floor((n_fft + 1) * bark_to_hz(barkpoints) / sr)
    # bin = np.array(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 38, 40, 42,
    #      44, 46, 48, 56, 64, 72, 80, 92, 104, 116, 128, 144, 160, 176, 192, 208, 232, 256])

    print(bin.shape)

    fbank = np.zeros([num_filters, n_fft // 2 + 1])
    for j in range(0, num_filters):
        print(j)
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


if __name__ == '__main__':
    fband = bark_filter_bank(56, 512, 16000, 100, 8000)
    print(fband.shape)
