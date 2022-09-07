import torch
import torch.nn as nn

# class CustomSRU(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.sru = SRU(*args, **kwargs)
#
#     def forward(self, x):
#         """
#
#         Args:
#             x: input
#
#         Shapes:
#             x: [B, T, F]
#             output: [B, T, F]
#         """
#         x = x.permute(1, 0, 2)
#         output_states, c_states = self.sru(x)
#         output_states = output_states.permute(1, 0, 2)
#         c_states = c_states.permute(1, 0, 2)
#         return output_states, c_states


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
    ):
        """
        Wrapper of conventional sequence models (LSTM or GRU)

        Args:
            input_size: input size.
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise, no linear layer.
            hidden_size: hidden size.
            num_layers:  number of layers.
            bidirectional: whether to use bidirectional RNN.
            sequence_model: LSTM | GRU.
            output_activate_function: Tanh | ReLU | ReLU6 | LeakyReLU | PReLU | None.
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "SRU":
            pass
            # self.sequence_model = CustomSRU(
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     num_layers=num_layers,
            #     bidirectional=bidirectional,
            #     highway_bias=-2
            # )
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

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"The shape of input is {x.shape}."
        self.sequence_model.flatten_parameters()

        x = x.permute(0, 2, 1)  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)

        if self.output_size:
            o = self.fc_output_layer(o)

        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
        return o


def _print_networks(nets: list):
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(
        f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
    )


if __name__ == "__main__":
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 257, 1000)
        model = SequenceModel(
            input_size=257,
            output_size=2,
            hidden_size=512,
            bidirectional=False,
            num_layers=3,
            sequence_model="LSTM",
        )

        start = datetime.datetime.now()
        opt = model(ipt)
        end = datetime.datetime.now()
        print(f"{end - start}")
        _print_networks([model])
