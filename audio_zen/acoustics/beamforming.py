import torch
import torch.nn as nn
from torch_complex import ComplexTensor
from torch_complex import functional as FC

from audio_zen.constant import EPSILON


def apply_crf_filter(
        cRM_filter: ComplexTensor,
        mix: ComplexTensor
) -> ComplexTensor:
    """
    Apply complex Ratio Filter

    Args:
        cRM_filter: complex Ratio Filter of shape [B, F, T, Filter_Delay]
        mix: mixture of shape [B, C, F, Filter_Delay,T]

    References:
        Generalized Spatio-Temporal RNN Beamformer for Target Speech Separation

    Returns:
        Shape of [B, C, F, T]
    """
    # [B, F, T, Filter_Delay] x [B, C, F, Filter_Delay,T] => [B, C, F, T]
    es = FC.einsum("bftd, bcfdt -> bcft", [cRM_filter.conj(), mix])
    return es


def get_power_spectral_density_matrix(
        complex_tensor: ComplexTensor
) -> ComplexTensor:
    """
    Cross-channel power spectral density (PSD) matrix

    Args:
        complex_tensor: speech or noise [..., F, C, T]

    Returns
        psd: [..., F, C, C]
    """
    # outer product: [..., C_1, T] x [..., C_2, T] => [..., T, C_1, C_2]
    return FC.einsum("...ct,...et->...tce", [complex_tensor, complex_tensor.conj()])


def get_power_spectral_density_matrix_with_mask_norm(
        mixture: ComplexTensor,
        mask: torch.Tensor,
        normalization: bool = True,
        eps: float = 1e-15,
) -> torch.complex64:
    """
    Cross-channel power spectral density (PSD) matrix with mask norm

    References:
        https://isca-speech.org/archive/Interspeech_2020/abstracts/1458.html

    Args:
        mixture: [B, F, C, T]
        mask: [B, F, C, T]
        normalization: use mask norm
        eps: eps

    Returns:
        psd: [B, F, C, C]
    """
    psd_y = FC.einsum("...ct, ...et ->...tce", [mixture, mixture.conj()])  # outer product: [B, F, C, T] => [B, F, T, C, C]

    # Averaging mask along the axis of channels: (B, F, C, T) -> (B, F, T)
    mask = mask.mean(dim=-2)

    # Normalized mask along the axis of times: (B, F, C, T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    psd = psd_y * mask[..., None, None]  # [B, F, T, C, C] => [B, F, T, C, C],
    psd = psd.sum(dim=-3)  # Time-invariant psd, [B, F, C, C]
    return psd


def trace(
        complex_matrix: ComplexTensor
) -> ComplexTensor:
    """
    Return trace of a complex matrices
    """
    mat_size = complex_matrix.size()
    diag_index = torch.eye(mat_size[-1], dtype=torch.bool, device=complex_matrix.device).expand(*mat_size)
    return complex_matrix.masked_select(diag_index).view(*mat_size[:-1]).sum(-1)


def mvdr_beamformer(
        noise_psd: ComplexTensor,
        steering_vector: ComplexTensor,
        eps=1e-15
) -> ComplexTensor:
    """
    Standard MVDR beamformer

    Args:
        noise_psd: time-averaged noise psd of shape [B, F, C, C]
        steering_vector: [B, F, C, 1]
        eps: eps

    Returns:
        beamformer_vector with shape of [B, F, C, 1]
    """
    batch_size, num_freqs, num_channels, _ = steering_vector.shape

    # More robust
    eye_matrix = torch.eye(num_channels, dtype=noise_psd.dtype, device=noise_psd.device)
    shape = [1 for _ in range(noise_psd.dim() - 2)] + [num_channels, num_channels]  # [1, 1, num_channels, num_channels]
    eye_matrix = eye_matrix.view(*shape)  # [1, 1, num_channels, num_channels]
    noise_psd += eps * eye_matrix

    noise_psd = noise_psd.reshape((batch_size * num_freqs, num_channels, num_channels))  # [B * F, C, C]

    # [B, C, F, 1] => [B, F, C, 1] => [B * F, C, 1]
    steering_vector = steering_vector.permute(0, 2, 1, 3).reshape((batch_size * num_freqs, num_channels, 1))

    psd_noise_inverse = noise_psd.inverse()
    w1 = FC.matmul(psd_noise_inverse, steering_vector)  # [B * F, C, 1]
    w2 = FC.matmul(steering_vector.conj().transpose(2, 1), psd_noise_inverse)  # [B * F, 1, C] * [B * F, C, C] * [B * F, C， 1] = [B * F, 1, 1]
    w2 = FC.matmul(w2, steering_vector)  # [B * F, 1, 1]

    w = w1 / w2  # [B * F, C, 1]

    # [B, F, C, 1]
    beamformer_vector = w.reshape((batch_size, num_freqs, num_channels, 1))

    return beamformer_vector


def pmwf_mvdr(
        speech_psd: torch.complex64,
        noise_psd: torch.complex64,
        reference_vector: torch.Tensor,
        eps: float = 1e-15
) -> torch.complex64:
    """
    PWMF MVDR

    Args:
        speech_psd: [B, F, C, C]
        noise_psd: [B, F, C, C]
        reference_vector: [B, C]
        eps: eps

    References:
        https://ieeexplore.ieee.org/document/5089420

    Returns:
        beamformer vector with shape of [B, F, C]
    """
    num_channels = noise_psd.size(-1)

    # Add eps
    eye = torch.eye(num_channels, dtype=noise_psd.dtype, device=noise_psd.device)
    shape = [1 for _ in range(noise_psd.dim() - 2)] + [num_channels, num_channels]  # [1, 1, 6, 6]
    eye = eye.view(*shape)  # [1, 1, 6, 6]
    noise_psd += eps * eye  # speech?

    inverse_noise_psd = noise_psd.inverse()
    inverse_noise_psd_speech_psd = inverse_noise_psd @ speech_psd
    trace_inverse_noise_psd_speech_psd = FC.trace(inverse_noise_psd_speech_psd) + eps

    ws = inverse_noise_psd_speech_psd / (trace_inverse_noise_psd_speech_psd[..., None, None] + eps)  # [B, F, C, C]
    beamformer_vector = FC.einsum("...fec, ...c -> ...fe", [ws, reference_vector])  # [B, F, C]
    return beamformer_vector


def apply_beamforming_vector(
        beamforming_vector: ComplexTensor,
        mixture: ComplexTensor
) -> ComplexTensor:
    """Apply beamforming weights at frame level

    Args:
        beamforming_vector: beamforming weighted vector with shape of [..., C]
        mixture: mixture of shape [..., C, T]

    Notes:
        There's no relationship between frequencies.

    Returns:
        [..., T]
    """
    # [..., C] x [..., C, T] => [..., T]
    es = FC.einsum("bftc, bfct -> bft", [beamforming_vector.conj(), mixture])
    return es


class MVDRBeamformer(nn.Module):
    """
    MVDR (Minimum Variance Distortionless Response) beamformer
    """

    def __init__(self, use_mask_norm: bool = True, eps: float = EPSILON):
        super().__init__()
        self.use_mask_norm = use_mask_norm
        self.eps = eps

    @staticmethod
    def stabilize_complex_number(complex_matrix: ComplexTensor, eps: float = EPSILON):
        return ComplexTensor(complex_matrix.real, complex_matrix.imag + torch.tensor(eps))

    def _derive_weight(self, speech_psd: ComplexTensor, noise_psd: ComplexTensor, reference_vector: torch.Tensor,
                       eps: float = 1e-5) -> ComplexTensor:
        """
        Derive MVDR beamformer

        Args:
            speech_psd: [B, F, C, C]
            noise_psd: [B, F, C, C]
            reference_vector: [B x C], reference selection vector
            eps:

        Return:
            [B, C, F]

        Examples:
            >>> B = 2
            >>> C = 8
            >>> F = 257
            >>> T = 200
            >>> sm = torch.rand(B, F, T)
            >>> nm = torch.rand(B, F, T)
            >>> c = ComplexTensor(torch.rand(B, C, F, T), torch.rand(B, C, F, T))
            >>> mvdr_beamformer = MVDRBeamformer()
            >>> output = mvdr_beamformer(sm, nm, c)
            >>> print(output.shape)
        """
        _, _, _, num_channels = noise_psd.shape

        identity_matrix = torch.eye(num_channels, device=noise_psd.device, dtype=noise_psd.dtype)
        noise_psd = noise_psd + identity_matrix * eps

        # [B, F, C, C]
        noise_psd_inverse = self.stabilize_complex_number(noise_psd.inverse())

        # [B, F, C, C]
        # einsum("...ij,...jk->...ik", Rn_inv, Rs)
        noise_psd_inverse_speech_psd = noise_psd_inverse @ speech_psd
        # [B, F]
        trace_noise_psd_inverse_speech_psd = trace(noise_psd_inverse_speech_psd) + eps
        # [B, F, C]
        # einsum("...fnc,...c->...fn", Rn_inv_Rs, u)
        noise_psd_inverse_speech_psd_u = (noise_psd_inverse_speech_psd @ reference_vector[:, None, :, None]).sum(-1)
        # [B, F, C]
        weight = noise_psd_inverse_speech_psd_u / trace_noise_psd_inverse_speech_psd[..., None]
        # [B, C ,F]
        return weight.transpose(1, 2)

    @staticmethod
    def mask_norm(mask: torch.Tensor) -> torch.Tensor:
        max_abs = torch.norm(mask, float("inf"), dim=1, keepdim=True)
        mask = mask / (max_abs + EPSILON)
        return mask

    @staticmethod
    def estimate_psd(mask: torch.Tensor, complex_matrix: ComplexTensor, eps: float = 1e-5) -> ComplexTensor:
        """
        Power Spectral Density Covariance (PSD) estimation

        Args:
            mask: [B, F, T], TF-masks (real)
            complex_matrix: [B, C, F, T], complex-valued matrix of short-term Fourier transform coefficients
            eps:

        Return:
            [B, F, C, C]
        """
        # [B, C, F, T] => [B, F, C, T]
        complex_matrix = complex_matrix.transpose(1, 2)
        # [B, F, T] => [B, F, 1, T]
        mask = mask.unsqueeze(-2)
        # [B, F, C, C]: einsum("...it,...jt->...ij", spec * mask, spec.conj())
        nominator = (complex_matrix * mask) @ complex_matrix.conj_transpose(-1, -2)
        # [B, F, 1, T] => [B, F, 1, 1]
        denominator = torch.clamp(mask.sum(-1, keepdims=True), min=eps)
        # [B, F, C, C]
        psd = nominator / denominator
        # stabilize
        return ComplexTensor(psd.real, psd.imag + torch.tensor(eps))

    def forward(self, speech_mask: torch.Tensor, noise_mask: torch.Tensor, complex_matrix: ComplexTensor) -> ComplexTensor:
        """
        Args:
            speech_mask: [B, F, T], real-valued speech T-F mask
            noise_mask: [B, F, T], real-valued noise T-F mask
            complex_matrix: [B x C x F x T], noisy complex spectrogram

        Return:
            [B, F, T], enhanced complex spectrogram
        """
        batch_size, num_channels, _, _ = complex_matrix.shape

        # B x F x T
        if self.use_mask_norm:
            speech_mask = self.mask_norm(speech_mask)
            noise_mask = self.mask_norm(noise_mask)

        # [B, F, C, C]
        speech_psd = self.estimate_psd(speech_mask, complex_matrix)
        noise_psd = self.estimate_psd(noise_mask, complex_matrix)
        speech_psd = ComplexTensor(speech_psd.real, speech_psd.imag + self.eps)
        noise_psd = ComplexTensor(noise_psd.real, noise_psd.imag + self.eps)

        # [B, C]
        reference_vector = torch.zeros((batch_size, num_channels), device=noise_psd.device, dtype=noise_psd.dtype)
        reference_vector[:, 0].fill_(1)

        # [B, C, F]
        weight = self._derive_weight(speech_psd, noise_psd, reference_vector, eps=self.eps)

        # [B, F, T]
        filtered_complex_matrix = self.apply_beamformer(weight, complex_matrix)

        return filtered_complex_matrix


def apply_beamformer_vector_at_utterance_level(
        beamforming_vector: ComplexTensor,
        mixture: ComplexTensor
) -> ComplexTensor:
    """Apply beamforming weights at utterance level

    Args:
        beamforming_vector: beamforming weighted vector with shape of [..., C]
        mixture: mixture of shape [..., C, T]

    Notes:
        For each frequency and omit time variation.

    Returns:
        [B, F, T]
    """
    return FC.einsum("...c, ...ct -> ...t", [beamforming_vector.conj(), mixture])


if __name__ == '__main__':
    torch.manual_seed(1)
    mixture = ComplexTensor(torch.rand(2, 257, 6, 200), torch.rand(2, 257, 6, 200))
    mask = torch.rand(2, 257, 6, 200)
    clean = ComplexTensor(torch.rand(2, 257, 6, 200), torch.rand(2, 257, 6, 200))
    noise = ComplexTensor(torch.rand(2, 257, 6, 200), torch.rand(2, 257, 6, 200))
    reference_vector = torch.tensor([[1., 0, 0, 0, 0, 0], [0, 1, 0., 0, 0, 0]])

    psd_n = get_power_spectral_density_matrix_with_mask_norm(noise, mask=mask, normalization=True)
    psd_s = get_power_spectral_density_matrix_with_mask_norm(clean, mask=mask, normalization=True)

    mvdr_beamformer_vector = pmwf_mvdr(psd_s, psd_n, reference_vector)
    enhanced = apply_beamformer_vector_at_utterance_level(mvdr_beamformer_vector, mixture)

    print(enhanced.shape)
