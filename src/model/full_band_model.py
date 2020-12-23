import torch
from torch.nn import functional

from common.model import BaseModel
from model.module.sequence import SequenceModel


class Model(BaseModel):
    def __init__(self,
                 n_freqs,
                 look_ahead,
                 sequence_model,
                 output_activate_function,
                 hidden_size,
                 weight_init=True,
                 use_offline_laplace_norm=True,
                 use_offline_gaussian_norm=False,
                 use_cumulative_norm=False,
                 use_forgetting_norm=False,
                 use_hybrid_norm=False
                 ):
        """
        Input: [B, 1, F, T]
        Output: [B, 2, F, T]

        Args:
            n_freqs:
            look_ahead:
            sequence_model:
            output_activate_function:
            hidden_size:
        """
        super().__init__()
        self.fband_model = SequenceModel(
            input_size=n_freqs,
            output_size=2 * n_freqs,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function
        )

        self.look_ahead = look_ahead
        self.use_offline_laplace_norm = use_offline_laplace_norm
        self.use_offline_gaussian_norm = use_offline_gaussian_norm
        self.use_cumulative_norm = use_cumulative_norm
        self.use_forgetting_norm = use_forgetting_norm
        self.use_hybrid_norm = use_hybrid_norm

        assert (use_hybrid_norm + use_forgetting_norm + use_cumulative_norm + use_offline_laplace_norm + use_offline_gaussian_norm) == 1, \
            "Only Supports one Norm method."

        if weight_init:
            print("Initializing Model...")
            self.apply(self.weight_init)

    def forward(self, input):
        """
        Args:
            input: [B, 1, F, T]

        Returns:
            model_ipt: [B, 2, F, T]
        """
        assert input.dim() == 4
        # Pad look ahead
        input = functional.pad(input, [0, self.look_ahead])
        batch_size, n_channels, n_freqs, n_frames = input.size()
        assert n_channels == 1, f"{self.__class__.__name__} takes mag feature as inputs."

        """=== === === Full-Band LSTM Model === === ==="""
        # [B, 1, F, T] => [B, F, T] => [B, 1, F, T]
        if self.use_offline_laplace_norm:
            fband_mu = torch.mean(input, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)  # 语谱图算一个均值
            fband_input = input / (fband_mu + 1e-10)
        elif self.use_offline_gaussian_norm:
            fband_mu = torch.mean(input, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)  # 语谱图算一个均值
            fband_std = torch.std(input, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)
            fband_input = (input - fband_mu) / (fband_std + 1e-10)
        elif self.use_cumulative_norm:
            fband_input = self.cumulative_norm(input)
        elif self.use_forgetting_norm:
            fband_input = self.forgetting_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        elif self.use_hybrid_norm:
            fband_input = self.hybrid_norm(input.reshape(batch_size, n_channels * n_freqs, n_frames), 192)
            fband_input.reshape(batch_size, n_channels, n_freqs, n_frames)
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_norm, cumulative_norm, forgetting_norm.")

        fband_input = fband_input.reshape(batch_size, n_channels * n_freqs, n_frames)
        fband_output = self.fband_model(fband_input)
        fband_output = fband_output.reshape(batch_size, 2, n_freqs, n_frames)

        output = fband_output[:, :, :, self.look_ahead:]
        return output


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 1, 161, 100)
        model = Model(
            n_freqs=161,
            look_ahead=1,
            sequence_model="LSTM",
            output_activate_function=None,
            hidden_size=512,
        )

        a = datetime.datetime.now()
        print(model(ipt).min())
        print(model(ipt).shape)
        b = datetime.datetime.now()
        print(f"{b - a}")
