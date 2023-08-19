"""
This is an enhanced version of the original FullSubNet model, incorporating a refined
approach to processing. Specifically, it employs a finer-to-coarser processing strategy,
whereby the lower subband section undergoes more detailed processing, while the higher
subband section undergoes more generalized processing. This design allows the model to
achieve a superior balance between performance and computational cost. Consequently,
this version of the FullSubNet model is well-suited for applications involving higher
sampling rates, such as 24kHz and 48kHz.

Check the bottom of this file for an example of how to use this model.

prerequisite:
    install pytorch according to your cuda version.
    pip install einops torchinfo
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional

EPSILON = np.finfo(np.float32).eps


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="GRU",
        output_activate_function="Tanh",
        num_groups=4,
        mogrify_steps=5,
        dropout=0.0,
    ):
        super().__init__()
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function
        self.output_size = output_size

        # only for custom_lstm and are needed to clean up
        self.sequence_model_name = sequence_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"Shape is {x.shape}."
        batch_size, _, _ = x.shape

        if self.sequence_model_name == "LayerNormLSTM":
            states = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]
        else:
            states = None

        x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
        o, _ = self.sequence_model(x, states)  # [T, B, F] => [T, B, F]

        if self.output_size:
            o = self.fc_output_layer(o)  # [T, B, F] => [T, B, F]

        if self.output_activate_function:
            o = self.activate_function(o)

        return o.permute(1, 2, 0).contiguous()  # [T, B, F] => [B, F, T]


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def offline_laplace_norm(input, return_mu=False):
        """Normalize the input with the utterance-level mean.

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]

        Notes:
            As mentioned in the paper, the offline normalization is used.
            Based on a bunch of experiments, the offline normalization have the same performance
            as the cumulative one and have a faster convergence than the cumulative one.
            Therefore, we use the offline normalization as the default normalization method.
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + EPSILON)

        if return_mu:
            return normed, mu
        else:
            return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """Normalize the input with the cumulative mean

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(
            batch_size * num_channels, 1, num_frames
        )

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)
        std = torch.std(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = (input - mu) / (std + EPSILON)
        return normed

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm


class SubBandSequenceWrapper(SequenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, subband_input):
        (
            batch_size,
            num_subband_units,
            num_channels,
            num_subband_freqs,
            num_frames,
        ) = subband_input.shape
        assert num_channels == 1

        output = subband_input.reshape(
            batch_size * num_subband_units, num_subband_freqs, num_frames
        )
        output = super().forward(output)

        # [B, N, C, 2, center, T]
        output = output.reshape(batch_size, num_subband_units, 2, -1, num_frames)

        # [B, 2, N, center, T]
        output = output.permute(0, 2, 1, 3, 4).contiguous()

        # [B, C, N * F_subband_out, T]
        output = output.reshape(batch_size, 2, -1, num_frames)

        return output


class SubbandModel(BaseModel):
    def __init__(
        self,
        freq_cutoffs,
        sb_num_center_freqs,
        sb_num_neighbor_freqs,
        fb_num_center_freqs,
        fb_num_neighbor_freqs,
        sequence_model,
        hidden_size,
        activate_function=False,
        norm_type="offline_laplace_norm",
    ):
        """Subband model.

        The subband model is a sequence model that takes the subband input and the fullband
        input as the input. The subband input is the subband sequence that is extracted from
        the noisy input. The subband sequence is extracted by unfolding the frequency axis.
        The fullband input is the fullband sequence that is extracted from the fullband
        output. The fullband sequence is also extracted by unfolding the frequency axis.

        For optimize the computational cost, the subband model has a finer-to-coarser
        processing. For the lower subband section, it gets more fine-grained processing.
        For the higher subband section, it gets more coarse-grained processing. In this way,
        the model can get a better trade-off between the performance and the computational
        cost.
        """
        super().__init__()

        sb_models = []
        for (
            sb_num_center_freq,
            sb_num_neighbor_freq,
            fb_num_center_freq,
            fb_num_neighbor_freq,
        ) in zip(
            sb_num_center_freqs,
            sb_num_neighbor_freqs,
            fb_num_center_freqs,
            fb_num_neighbor_freqs,
        ):
            sb_models.append(
                SubBandSequenceWrapper(
                    input_size=(sb_num_center_freq + sb_num_neighbor_freq * 2)
                    + (fb_num_center_freq + fb_num_neighbor_freq * 2),
                    output_size=sb_num_center_freq * 2,
                    hidden_size=hidden_size,
                    num_layers=2,
                    sequence_model=sequence_model,
                    bidirectional=False,
                    output_activate_function=activate_function,
                )
            )

        self.sb_models = nn.ModuleList(sb_models)
        self.freq_cutoffs = freq_cutoffs
        self.sb_num_center_freqs = sb_num_center_freqs
        self.sb_num_neighbor_freqs = sb_num_neighbor_freqs
        self.fb_num_center_freqs = fb_num_center_freqs
        self.fb_num_neighbor_freqs = fb_num_neighbor_freqs

        self.norm = self.norm_wrapper(norm_type)

    def _freq_unfold(
        self,
        input: torch.Tensor,
        lower_cutoff_freq=0,
        upper_cutoff_freq=20,
        num_center_freqs=1,
        num_neighbor_freqs=15,
    ):
        """Unfold frequency axis.

        Args:
            input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames).
            cutoff_freq_lower: lower cutoff frequency.
            cutoff_freq_higher: higher cutoff frequency.
            num_center_freqs: number of center frequencies.
            num_neighbor_freqs: number of neighbor frequencies.

        Returns:
            [batch_size, num_subband_units, num_channels, num_subband_freqs, num_frames]

        Note:
            We assume that the num_neighbor_freqs should less than the minimum subband intervel.
        """
        batch_size, num_channels, num_freqs, num_frames = input.shape
        assert num_channels == 1, "Only mono audio is supported."

        if (upper_cutoff_freq - lower_cutoff_freq) % num_center_freqs != 0:
            raise ValueError(
                f"The number of center frequencies should be divisible by the subband freqency interval. "
                f"Got {num_center_freqs=}, {upper_cutoff_freq=}, and {lower_cutoff_freq=}. "
                f"The subband freqency interval is {upper_cutoff_freq-lower_cutoff_freq}."
            )

        # extract valid input with the shape of [batch_size, 1, num_freqs, num_frames]
        if lower_cutoff_freq == 0:
            # lower = 0, upper = upper_cutoff_freq + num_neighbor_freqs
            valid_input = input[..., 0 : (upper_cutoff_freq + num_neighbor_freqs), :]
            valid_input = functional.pad(
                input=valid_input,
                pad=(0, 0, num_neighbor_freqs, 0),
                mode="reflect",
            )

        elif upper_cutoff_freq == num_freqs:
            # lower = lower_cutoff_freq - num_neighbor_freqs, upper = num_freqs
            valid_input = input[
                ..., lower_cutoff_freq - num_neighbor_freqs : num_freqs, :
            ]

            valid_input = functional.pad(
                input=valid_input,
                pad=(0, 0, 0, num_neighbor_freqs),
                mode="reflect",
            )
        else:
            # lower = lower_cutoff_freq - num_neighbor_freqs, upper = upper_cutoff_freq + num_neighbor_freqs
            valid_input = input[
                ...,
                lower_cutoff_freq
                - num_neighbor_freqs : upper_cutoff_freq
                + num_neighbor_freqs,
                :,
            ]

        # unfold
        # [B, C * kernel_size, N]
        subband_unit_width = num_center_freqs + num_neighbor_freqs * 2
        output = functional.unfold(
            input=valid_input,
            kernel_size=(subband_unit_width, num_frames),
            stride=(num_center_freqs, num_frames),
        )
        num_subband_units = output.shape[-1]

        output = output.reshape(
            batch_size,
            num_channels,
            subband_unit_width,
            num_frames,
            num_subband_units,
        )

        # [B, N, C, F_subband, T]
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    def forward(self, noisy_input, fb_output):
        """Forward pass.

        Args:
            input: magnitude spectrogram of shape (batch_size, 1, num_freqs, num_frames).
        """
        batch_size, num_channels, num_freqs, num_frames = noisy_input.size()
        assert num_channels == 1, "Only mono audio is supported."

        subband_output = []
        for sb_idx, sb_model in enumerate(self.sb_models):
            if sb_idx == 0:
                lower_cutoff_freq = 0
                upper_cutoff_freq = self.freq_cutoffs[0]
            elif sb_idx == len(self.sb_models) - 1:
                lower_cutoff_freq = self.freq_cutoffs[-1]
                upper_cutoff_freq = num_freqs
            else:
                lower_cutoff_freq = self.freq_cutoffs[sb_idx - 1]
                upper_cutoff_freq = self.freq_cutoffs[sb_idx]

            # unfold frequency axis
            # [B, N, C, F_subband, T]
            noisy_subband = self._freq_unfold(
                noisy_input,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.sb_num_center_freqs[sb_idx],
                self.sb_num_neighbor_freqs[sb_idx],
            )

            # [B, N, C, F_subband, T]
            fb_subband = self._freq_unfold(
                fb_output,
                lower_cutoff_freq,
                upper_cutoff_freq,
                self.fb_num_center_freqs[sb_idx],
                self.fb_num_neighbor_freqs[sb_idx],
            )

            sb_model_input = torch.cat([noisy_subband, fb_subband], dim=-2)
            sb_model_input = self.norm(sb_model_input)
            subband_output.append(sb_model(sb_model_input))

        # [B, C, F, T]
        output = torch.cat(subband_output, dim=-2)

        return output


class Model(BaseModel):
    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        win_length=512,
        fdrc=0.5,
        num_freqs=257,
        freq_cutoffs=[20, 80],
        sb_num_center_freqs=[1, 4, 8],
        sb_num_neighbor_freqs=[15, 15, 15],
        fb_num_center_freqs=[1, 4, 8],
        fb_num_neighbor_freqs=[15, 15, 15],
        fb_hidden_size=512,
        sb_hidden_size=384,
        sequence_model="LSTM",
        fb_output_activate_function=False,
        sb_output_activate_function=False,
        norm_type="offline_laplace_norm",
    ):
        """FullSubNet model.

        This is an improved version of the original FullSubNet model. The main difference
        is that the subband model has a finer-to-coarser processing. For the lower subband
        section, it gets more fine-grained processing. For the higher subband section, it
        gets more coarse-grained processing. In this way, the model can get a better trade-off
        between the performance and the computational cost.

        Args:
            n_fft: number of fft points. Defaults to 512 for 16kHz.
            hop_length: defaults to 128 for 16kHz.
            win_length: defaults to 512 for 16kHz.
            fdrc: frequency dynamic range compression. Defaults to 0.5.
            num_freqs: number of frequency bins. Defaults to 257 for 16kHz.
            freq_cutoffs: cutoff frequencies for subband sections. Defaults to [32, 128],
                It means that the first subband section is [0, 32], the second subband
                section is [32, 128], and the third subband section is [128, 257]. For
                lower subband section, it gets more fine-grained processing.
            sb_num_center_freqs: number of center frequencies for each subband section.
                Defaults to [1, 4, 8]. It means that the first subband section has 1 center
                frequency, the second subband section has 4 center frequencies, and the third
                subband section has 8 center frequencies. The processing gets more coarse-grained.
            sb_num_neighbor_freqs: number of neighbor frequencies for each subband section.
                Defaults to [15, 15, 15].
            fb_num_center_freqs: similar to sb_num_center_freqs, but for fullband.
            fb_num_neighbor_freqs: similar to sb_num_neighbor_freqs, but for fullband.
            fb_hidden_size: hidden size for fullband model. Defaults to 512.
            sb_hidden_size: hidden size for subband model. Defaults to 384.
            sequence_model: sequence model type. Defaults to "LSTM".
            fb_output_activate_function: fullband output activation function. Defaults to False.
            sb_output_activate_function: subband output activation function. Defaults to False.
            norm_type: normalization type. Defaults to "offline_laplace_norm". The cumulative
                version is "cumulative_laplace_norm". They are the same in performance.

        Notes:
            The default settings are for 16kHz sampling rate. If you want to use it for
            24kHz, you may adjust the parameters accordingly. The basic principle is that
            the lower subband section has more fine-grained processing, and the higher
            subband section has more coarse-grained processing.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fdrc = fdrc

        self.fb_model = SequenceModel(
            input_size=num_freqs - 1,  # remove the last one for easier processing
            output_size=num_freqs - 1,
            hidden_size=fb_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function,
        )

        self.sb_model = SubbandModel(
            freq_cutoffs=freq_cutoffs,
            sb_num_center_freqs=sb_num_center_freqs,
            sb_num_neighbor_freqs=sb_num_neighbor_freqs,
            fb_num_center_freqs=fb_num_center_freqs,
            fb_num_neighbor_freqs=fb_num_neighbor_freqs,
            hidden_size=sb_hidden_size,
            sequence_model=sequence_model,
            activate_function=sb_output_activate_function,
        )

        self.norm = self.norm_wrapper(norm_type)

    def forward(self, y):
        ndim = y.dim()
        assert ndim in (2, 3), "Input must be 2D (B, T) or 3D tensor (B, 1, T)"

        if ndim == 3:
            assert y.size(1) == 1, "Input must be 2D (B, T) or 3D tensor (B, 1, T)"
            y = y.squeeze(1)

        # [B, F, T]
        complex_stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )

        complex_stft_view_real = torch.view_as_real(complex_stft)  # [B, F, T, 2]

        noisy_mag = torch.abs(complex_stft.unsqueeze(1))  # [B, 1, F, T]

        # ================== Fullband ==================
        noisy_mag = noisy_mag**self.fdrc  # frequency dynamic range compression
        noisy_mag = noisy_mag[..., :-1, :]  # remove the last one
        fb_input = rearrange(self.norm(noisy_mag), "b c f t -> b (c f) t")
        fb_output = self.fb_model(fb_input)  # [B, F, T]
        fb_output = rearrange(fb_output, "b f t -> b 1 f t")

        # ================== Subband ==================
        cRM = self.sb_model(noisy_mag, fb_output)  # [B, 2, F, T]
        cRM = functional.pad(cRM, (0, 0, 0, 1), mode="constant", value=0.0)

        # ================== Masking ==================
        complex_stft_view_real = rearrange(complex_stft_view_real, "b f t c -> b c f t")
        enhanced_spec = cRM * complex_stft_view_real  # [B, 2, F, T]

        # ================== ISTFT ==================
        enhanced_complex = torch.complex(
            enhanced_spec[:, 0, ...], enhanced_spec[:, 1, ...]
        )
        enhanced_y = torch.istft(
            enhanced_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            length=y.size(-1),
        )
        enhanced_y = enhanced_y.unsqueeze(1)  # [B, 1, T]
        return enhanced_y


if __name__ == "__main__":
    from torchinfo import summary

    # Test the model for 16kHz data
    model = Model()
    noisy_y = torch.rand(1, 16000)
    summary(model, input_data=(noisy_y,), device="cpu")

    # Test the model for 48kHz data
    high_sampling_rate_model = Model(
        n_fft=960,
        hop_length=480,
        win_length=960,
        fdrc=0.5,
        num_freqs=481,
        freq_cutoffs=[20, 120, 240],
        sb_num_center_freqs=[1, 4, 20, 60],
        sb_num_neighbor_freqs=[15, 15, 15, 15],
        fb_num_center_freqs=[1, 4, 20, 60],
        fb_num_neighbor_freqs=[15, 15, 15, 15],
        fb_hidden_size=512,
        sb_hidden_size=384,
        sequence_model="LSTM",
        fb_output_activate_function=False,
        sb_output_activate_function=False,
        norm_type="offline_laplace_norm",
    )
    noisy_y = torch.rand(1, 48000)
    summary(high_sampling_rate_model, input_data=(noisy_y,), device="cpu")
    summary(high_sampling_rate_model, input_data=(noisy_y,), device="cpu")
