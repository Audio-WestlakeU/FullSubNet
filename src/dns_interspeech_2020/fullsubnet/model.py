import torch
from torch.nn import functional

from audio_zen.acoustic.feature import drop_sub_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel


class Model(BaseModel):
    def __init__(self,
                 num_freqs,
                 subband_num_neighbors,
                 fullband_num_neighbors,
                 look_ahead,
                 sequence_model,
                 fullband_output_activate_function,
                 subband_output_activate_function,
                 fullband_model_hidden_size,
                 subband_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 weight_init=True,
                 num_sub_batches=3):
        """
        FullSubNet model (cIRM Model)

        Args:
            num_freqs: Frequency dim of the input
            subband_num_neighbors: Number of the neighbor frequencies in each side
            fullband_num_neighbors:
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fullband_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fullband_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fullband_output_activate_function
        )

        self.subband_model = SequenceModel(
            input_size=(subband_num_neighbors * 2 + 1) + 1 + 2,
            output_size=2,
            hidden_size=subband_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=subband_output_activate_function
        )

        self.subband_num_neighbor = subband_num_neighbors
        self.fullband_num_neighbors = fullband_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_sub_batches = num_sub_batches

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: [B, 1, F, T], noisy magnitude spectrogram

        Returns:
            [B, 2, F, T], the real part and imag part of the enhanced spectrogram
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband
        fullband_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fullband_output = self.fullband_model(fullband_input).reshape(batch_size, 1, num_freqs, num_frames)

        if self.fullband_num_neighbors == 0:
            fullband_output_unfolded = fullband_output.permute(0, 2, 1, 3)
            fullband_output_unfolded = fullband_output_unfolded.reshape(batch_size * num_freqs, 1, num_frames)
        else:
            # unfold, [B, N=F, C, F_f, T]
            fullband_output_unfolded = self.unfold(fullband_output, n_neighbor=self.fullband_num_neighbors)
            fullband_output_unfolded = fullband_output_unfolded.reshape(batch_size * num_freqs, self.fullband_num_neighbors * 2 + 1, num_frames)

        # unfold, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, n_neighbor=self.subband_num_neighbor)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size * num_freqs, self.subband_num_neighbor * 2 + 1, num_frames)

        # concat, [B * F, (F_s + F_f), T]
        subband_input = torch.cat([noisy_mag_unfolded, fullband_output_unfolded], dim=1)
        subband_input = self.norm(subband_input)

        # Speed up training without significant performance degradation
        # This part of the content will be updated in the paper later
        if batch_size > 1:
            subband_input = subband_input.reshape(
                batch_size,
                num_freqs,
                self.subband_num_neighbor * 2 + 1 + self.fullband_num_neighbors * 2 + 1,
                num_frames
            )
            subband_input = drop_sub_band(subband_input.permute(0, 2, 1, 3), num_sub_batches=self.num_sub_batches)
            n_freqs = subband_input.shape[2]
            subband_input = subband_input.permute(0, 2, 1, 3).reshape(-1, self.subband_num_neighbor * 2 + 1 + 1, num_frames)

        # [B * F, (F_s + 1), T] => [B * F, 2, T] => [B, F, 2, T]
        sband_mask = self.subband_model(subband_input)
        sband_mask = sband_mask.reshape(batch_size, n_freqs, 2, n_frames).permute(0, 2, 1, 3).contiguous()

        output = sband_mask[:, :, :, self.look_ahead:]
        return output


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        model = Model(
            subband_num_neighbors=15,
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fullband_output_activate_function="ReLU",
            subband_output_activate_function=None,
            fullband_model_hidden_size=512,
            subband_model_hidden_size=384,
            use_offline_norm=True,
            weight_init=False,
            use_cumulative_norm=False,
            use_forgetting_norm=False,
            use_hybrid_norm=False,
        )
        ipt = torch.rand(3, 800)  # 1.6s
        ipt_len = ipt.shape[-1]
        # 1000 frames (16s) - 5.65s (35.31%，纯模型) - 5.78s
        # 500 frames (8s) - 3.05s (38.12%，纯模型) - 3.04s
        # 200 frames (3.2s) - 1.19s (37.19%，纯模型) - 1.20s
        # 100 frames (1.6s) - 0.62s (38.75%，纯模型) - 0.65s
        start = datetime.datetime.now()

        complex_tensor = torch.stft(ipt, n_fft=512, hop_length=256)
        mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0).unsqueeze(1)
        print(f"STFT: {datetime.datetime.now() - start}, {mag.shape}")

        enhanced_complex_tensor = model(mag).detach().permute(0, 2, 3, 1)
        print(enhanced_complex_tensor.shape)
        print(f"Model Inference: {datetime.datetime.now() - start}")

        enhanced = torch.istft(enhanced_complex_tensor, 512, 256, length=ipt_len)
        print(f"iSTFT: {datetime.datetime.now() - start}")

        print(f"{datetime.datetime.now() - start}")
