import torch
import torch.nn as nn


def cumulative_norm(input):
    eps = 1e-10

    # [B, C, F, T]
    batch_size, n_channels, n_freqs, n_frames = input.size()
    device = input.device
    data_type = input.dtype

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
    x = x.reshape(batch_size, n_channels, n_freqs, n_frames)

    return x


class CumulativeMagSpectralNorm(nn.Module):
    def __init__(self, cumulative=False, use_mid_freq_mu=False):
        """

        Args:
            cumulative: 是否采用累积的方式计算 mu
            use_mid_freq_mu: 仅采用中心频率的 mu 来代替全局 mu

        Notes:
            先算均值再累加 等同于 先累加再算均值

        """
        super().__init__()
        self.eps = 1e-6
        self.cumulative = cumulative
        self.use_mid_freq_mu = use_mid_freq_mu

    def forward(self, input):
        assert input.ndim == 4, f"{self.__name__} only support 4D input."
        batch_size, n_channels, n_freqs, n_frames = input.size()
        device = input.device
        data_type = input.dtype

        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)

        if self.use_mid_freq_mu:
            step_sum = input[:, int(n_freqs // 2 - 1), :]  # [B * C, F, T] => [B * C, T]
        else:
            step_sum = torch.mean(input, dim=1)  # [B * C, F, T] => [B * C, T]

        if self.cumulative:
            cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
            entry_count = torch.arange(1, n_frames + 1, dtype=data_type, device=device)
            entry_count = entry_count.reshape(1, n_frames)  # [1, T]
            entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

            mu = cumulative_sum / entry_count  # [B * C, T]
            mu = mu.reshape(batch_size * n_channels, 1, n_frames)
        else:
            mu = torch.mean(step_sum, dim=-1)  # [B * C]
            mu = mu.reshape(batch_size * n_channels, 1, 1)  # [B * C, 1, 1]

        input_normed = input / (mu + self.eps)
        input_normed = input_normed.reshape(batch_size, n_channels, n_freqs, n_frames)
        return input_normed


if __name__ == '__main__':
    a = torch.rand(2, 1, 160, 200)
    ln = CumulativeMagSpectralNorm(cumulative=False, use_mid_freq_mu=False)
    ln_1 = CumulativeMagSpectralNorm(cumulative=True, use_mid_freq_mu=False)
    ln_2 = CumulativeMagSpectralNorm(cumulative=False, use_mid_freq_mu=False)
    ln_3 = CumulativeMagSpectralNorm(cumulative=True, use_mid_freq_mu=False)
    print(ln(a).mean())
    print(ln_1(a).mean())
    print(ln_2(a).mean())
    print(ln_3(a).mean())
