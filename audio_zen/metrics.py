import numpy as np
from pesq import pesq
from pystoi.stoi import stoi


def SI_SDR(reference, estimation, sr=16000):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)。

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References:
        SDR- Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference**2, axis=-1, keepdims=True)

    optimal_scaling = (
        np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    )

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection**2, axis=-1) / np.sum(noise**2, axis=-1)
    return 10 * np.log10(ratio)


def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False)


def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb")


def NB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "nb")


# Only registered metric can be used.
REGISTERED_METRICS = {
    "SI_SDR": SI_SDR,
    "STOI": STOI,
    "WB_PESQ": WB_PESQ,
    "NB_PESQ": NB_PESQ,
}
