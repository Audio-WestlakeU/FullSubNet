import torch
from torch.nn import functional

from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel


class Model(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold the output of the fullband model, [B, N=F, C, F_f, T]
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy input, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation. These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        return output


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        model = Model(
            sb_num_neighbors=15,
            fb_num_neighbors=0,
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_output_activate_function="ReLU",
            sb_output_activate_function=None,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            weight_init=False,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
        )
        # ipt = torch.rand(3, 800)  # 1.6s
        # ipt_len = ipt.shape[-1]
        # # 1000 frames (16s) - 5.65s (35.31%，纯模型) - 5.78s
        # # 500 frames (8s) - 3.05s (38.12%，纯模型) - 3.04s
        # # 200 frames (3.2s) - 1.19s (37.19%，纯模型) - 1.20s
        # # 100 frames (1.6s) - 0.62s (38.75%，纯模型) - 0.65s
        # start = datetime.datetime.now()
        #
        # complex_tensor = torch.stft(ipt, n_fft=512, hop_length=256)
        # mag = (complex_tensor.pow(2.).sum(-1) + 1e-8).pow(0.5 * 1.0).unsqueeze(1)
        # print(f"STFT: {datetime.datetime.now() - start}, {mag.shape}")
        #
        # enhanced_complex_tensor = model(mag).detach().permute(0, 2, 3, 1)
        # print(enhanced_complex_tensor.shape)
        # print(f"Model Inference: {datetime.datetime.now() - start}")
        #
        # enhanced = torch.istft(enhanced_complex_tensor, 512, 256, length=ipt_len)
        # print(f"iSTFT: {datetime.datetime.now() - start}")
        #
        # print(f"{datetime.datetime.now() - start}")
        ipt = torch.rand(3, 1, 257, 200)
        print(model(ipt).shape)
