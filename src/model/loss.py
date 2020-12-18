import torch
import torch.nn as nn


l1_loss = torch.nn.L1Loss
mse_loss = torch.nn.MSELoss


def si_sdr_loss():
    return SISDRLoss()


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals. Used in end-to-end networks.
    This is essentially a batch PyTorch version of the function
    ``nussl.evaluation.bss_eval.scale_bss_eval`` and can be used to compute
    SI-SDR or SNR.

    Args:
        scaling (bool, optional): Whether to use scale-invariant (True) or
          signal-to-noise ratio (False). Defaults to True.
        return_scaling (bool, optional): Whether to only return the scaling
          factor that the estimate gets scaled by relative to the reference.
          This is just for monitoring this value during training, don't actually
          train with it! Defaults to False.
        reduction (str, optional): How to reduce across the batch (either 'mean',
          'sum', or none). Defaults to 'mean'.
        zero_mean (bool, optional): Zero mean the references and estimates before
          computing the loss. Defaults to True.
    """
    def __init__(self, scaling=True, return_scaling=False, reduction='mean',
                 zero_mean=True):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.return_scaling = return_scaling
        super().__init__()

    def forward(self, references, estimates):
        eps = 1e-8
        references = references.unsqueeze(-1)
        estimates = estimates.unsqueeze(-1)
        # num_batch, num_samples, num_sources
        _shape = references.shape
        references = references.view(-1, _shape[-2], _shape[-1])
        estimates = estimates.view(-1, _shape[-2], _shape[-1])
        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling else 1)

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = 10 * torch.log10(signal / noise + eps)

        if self.reduction == 'mean':
            sdr = sdr.mean()
        elif self.reduction == 'sum':
            sdr = sdr.sum()
        if self.return_scaling:
            return scale
        # go negative so it's a loss
        return -sdr


if __name__ == '__main__':
    a = torch.rand(2, 15000)
    b = torch.rand(2, 15000)

    loss = si_sdr_loss()
    print(loss(a, b))
