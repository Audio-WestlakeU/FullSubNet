from torch_complex import ComplexTensor
from torch_complex import functional as FC


def apply_crf_filter(cRM_filter: ComplexTensor, mix: ComplexTensor) -> ComplexTensor:
    """
    Apply complex Ratio Filter

    Args:
        cRM_filter: complex Ratio Filter
        mix: mixture

    Returns:
        [B, C, F, T]
    """
    # [B, F, T, Filter_delay] x [B, C, F, Filter_delay,T] => [B, C, F, T]
    es = FC.einsum("bftd, bcfdt -> bcft", [cRM_filter.conj(), mix])
    return es


def get_power_spectral_density_matrix(complex_tensor: ComplexTensor) -> ComplexTensor:
    """
    Cross-channel power spectral density (PSD) matrix

    Args:
        complex_tensor: [..., F, C, T]

    Returns
        psd: [..., F, C, C]
    """
    # outer product: [..., C_1, T] x [..., C_2, T] => [..., T, C_1, C_2]
    return FC.einsum("...ct,...et->...tce", [complex_tensor, complex_tensor.conj()])


def apply_beamforming_vector(beamforming_vector: ComplexTensor, mix: ComplexTensor) -> ComplexTensor:
    # [..., C] x [..., C, T] => [..., T]
    # There's no relationship between frequencies.
    es = FC.einsum("bftc, bfct -> bft", [beamforming_vector.conj(), mix])
    return es
