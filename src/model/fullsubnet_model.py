import torch
from torch.nn import functional

from common.model import BaseModel
from model.module.sequence import SequenceModel
from util.acoustic_utils import drop_sub_band


class Model(BaseModel):
    def __init__(self,
                 n_freqs,
                 n_neighbor,
                 look_ahead,
                 sequence_model,
                 fband_output_activate_function,
                 sband_output_activate_function,
                 fband_model_hidden_size,
                 sband_model_hidden_size,
                 bidirectional=False,
                 weight_init=True,
                 num_sub_batches=3,
                 use_offline_norm=True,
                 use_cumulative_norm=False,
                 use_forgetting_norm=False,
                 use_hybrid_norm=False,
                 ):
        """
        FullSubNet model

        Input: [B, 1, F, T]
        Output: [B, 2, F, T]

        Args:
            n_freqs: Frequency dim of the input
            n_neighbor: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fband_model = SequenceModel(
            input_size=n_freqs,
            output_size=n_freqs,
            hidden_size=fband_model_hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            sequence_model=sequence_model,
            output_activate_function=fband_output_activate_function
        )

        self.sband_model = SequenceModel(
            input_size=(n_neighbor * 2 + 1) + 1 + 2,
            output_size=2,
            hidden_size=sband_model_hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            sequence_model=sequence_model,
            output_activate_function=sband_output_activate_function
        )

        self.n_neighbor = n_neighbor
        self.look_ahead = look_ahead
        self.use_offline_norm = use_offline_norm
        self.use_cumulative_norm = use_cumulative_norm
        self.use_forgetting_norm = use_forgetting_norm
        self.use_hybrid_norm = use_hybrid_norm
        self.num_sub_batches = num_sub_batches

        assert (use_hybrid_norm + use_forgetting_norm + use_cumulative_norm + use_offline_norm) == 1, \
            "Only Supports one Norm method."

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, input):
        """
        Args:
            input: [B, 1, F, T]

        Returns:
            [B, 2, F, T]
        """
        assert input.dim() == 4
        # Pad look ahead
        input = functional.pad(input, [0, self.look_ahead])
        batch_size, n_channels, n_freqs, n_frames = input.size()
        assert n_channels == 1, f"{self.__class__.__name__} takes mag feature as inputs."

        """=== === === Full-Band sub Model === === ==="""
        if self.use_offline_norm:
            fband_mu = torch.mean(input, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)  # 语谱图算一个均值
            fband_input = input / (fband_mu + 1e-10)
        elif self.use_cumulative_norm:
            fband_input = self.cumulative_norm(input)
        elif self.use_forgetting_norm:
            fband_input = self.forgetting_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        elif self.use_hybrid_norm:
            fband_input = self.hybrid_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        else:
            raise NotImplementedError("You must set up a type of Norm. E.g., offline_norm, cumulative_norm, forgetting_norm.")

        # [B, 1, F, T] => [B, F, T] => [B, 1, F, T]
        fband_input = fband_input.reshape(batch_size, n_channels * n_freqs, n_frames)
        fband_output = self.fband_model(fband_input)
        fband_output = fband_output.reshape(batch_size, n_channels, n_freqs, n_frames)

        """=== === === Sub-Band sub Model === === ==="""
        # [B, 1, F, T] => unfold => [B, N=F, C, F_s, T] => [B * N, F_s, T]
        input_unfolded = self.unfold(input, n_neighbor=self.n_neighbor)
        fband_output_unfolded = self.unfold(fband_output, n_neighbor=1)

        input_unfolded = input_unfolded.reshape(batch_size * n_freqs, self.n_neighbor * 2 + 1, n_frames)
        fband_output_unfolded = fband_output_unfolded.reshape(batch_size * n_freqs, 2 + 1, n_frames)

        # [B * F, (F_s + 3), T]
        sband_input = torch.cat([input_unfolded, fband_output_unfolded], dim=1)

        if self.use_offline_norm:
            sband_mu = torch.mean(sband_input, dim=(1, 2)).reshape(batch_size * n_freqs, 1, 1)
            sband_input = sband_input / (sband_mu + 1e-10)
        elif self.use_cumulative_norm:
            sband_input = self.cumulative_norm(sband_input)
        elif self.use_forgetting_norm:
            sband_input = self.forgetting_norm(sband_input, 192)
        elif self.use_hybrid_norm:
            sband_input = self.hybrid_norm(sband_input, 192)
        else:
            raise NotImplementedError("You must set up a type of Norm. E.g., offline_norm, cumulative_norm, forgetting_norm.")

        # Speed up training without significant performance degradation
        # This part of the content will be updated in the paper later
        if batch_size > 1:
            sband_input = sband_input.reshape(batch_size, n_freqs, self.n_neighbor * 2 + 1 + 2 + 1, n_frames)
            sband_input = drop_sub_band(sband_input.permute(0, 2, 1, 3), num_sub_batches=self.num_sub_batches)
            n_freqs = sband_input.shape[2]
            sband_input = sband_input.permute(0, 2, 1, 3).reshape(-1, self.n_neighbor * 2 + 1 + 2 + 1, n_frames)

        # [B * F, (F_s + 1), T] => [B * F, 2, T] => [B, F, 2, T]
        sband_mask = self.sband_model(sband_input)
        sband_mask = sband_mask.reshape(batch_size, n_freqs, 2, n_frames).permute(0, 2, 1, 3).contiguous()

        output = sband_mask[:, :, :, self.look_ahead:]
        return output


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        model = Model(
            n_neighbor=15,
            n_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fband_output_activate_function="ReLU",
            sband_output_activate_function=None,
            fband_model_hidden_size=512,
            sband_model_hidden_size=384,
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
