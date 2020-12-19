import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pypesq import pesq as nb_pesq
from pystoi.stoi import stoi


def _scale_bss_eval(references, estimate, idx, compute_sir_sar=True):
    """
    Helper for scale_bss_eval to avoid infinite recursion loop.
    """
    source = references[..., idx]
    source_energy = (source ** 2).sum()

    alpha = (
            source @ estimate / source_energy
    )

    e_true = source
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    snr = 10 * np.log10(signal / noise)

    e_true = source * alpha
    e_res = estimate - e_true

    signal = (e_true ** 2).sum()
    noise = (e_res ** 2).sum()

    si_sdr = 10 * np.log10(signal / noise)

    srr = -10 * np.log10((1 - (1 / alpha)) ** 2)
    sd_sdr = snr + 10 * np.log10(alpha ** 2)

    si_sir = np.nan
    si_sar = np.nan

    if compute_sir_sar:
        references_projection = references.T @ references

        references_onto_residual = np.dot(references.transpose(), e_res)
        b = np.linalg.solve(references_projection, references_onto_residual)

        e_interf = np.dot(references, b)
        e_artif = e_res - e_interf

        si_sir = 10 * np.log10(signal / (e_interf ** 2).sum())
        si_sar = 10 * np.log10(signal / (e_artif ** 2).sum())

    return si_sdr, si_sir, si_sar, sd_sdr, snr, srr


def SDR(reference, estimation, sr=16000):
    sdr, _, _, _ = bss_eval_sources(reference[None, :], estimation[None, :])
    return sdr


def SI_SDR(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References
        SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False)


def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb")


def NB_PESQ(ref, est, sr=16000):
    return nb_pesq(ref, est, sr)


# Only registered metric can be used.
REGISTERED_METRICS = {
    "SI_SDR": SI_SDR,
    "STOI": STOI,
    "WB_PESQ": WB_PESQ,
    "NB_PESQ": NB_PESQ
}
