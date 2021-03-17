import torch
from torch.nn import functional

from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel


class Model(BaseModel):
    def __init__(
            self,
            num_freqs,
            hidden_size,
            sequence_model,
            output_activate_function,
            look_ahead,
            norm_type="offline_laplace_norm",
            weight_init=True,
    ):
        """
        Fullband Model (cIRM mask)

        Args:
            num_freqs:
            hidden_size:
            sequence_model:
            output_activate_function:
            look_ahead:
        """
        super().__init__()
        self.fullband_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs * 2,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=output_activate_function
        )

        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        if weight_init:
            print("Initializing model...")
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

        noisy_mag = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        output = self.fullband_model(noisy_mag).reshape(batch_size, 2, num_freqs, num_frames)

        return output[:, :, :, self.look_ahead:]


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 1, 161, 100)
        model = Model(
            num_freqs=161,
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
