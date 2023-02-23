import numpy as np
from numpy.typing import NDArray


def reverberation_time_shortening(
    rir: NDArray,
    original_T60: float,
    target_T60: float,
    sr: int = 16000,
    time_after_max: float = 0.002,
) -> tuple(NDArray, NDArray):
    """Shorten reverberation time of a RIR.

    See this paper for more details:
        Speech Dereverberation With a Reverberation Time Shortening Target
        https://arxiv.org/abs/2204.08765

    Args:
        rir: given RIR.
        original_T60: the rt60 of the given RIR.
        target_T60: the target rt60.
        sr: sampling rate. Defaults to 16000.
        time_after_max: the time after the maximum of the RIR. Defaults to 0.002.

    Returns:
        The shortened RIR and the window.

    Cite:
        @article{zhou2022single,
            title={Single-Channel Speech Dereverberation using Subband Network with A Reverberation Time Shortening Target},
            author={Zhou, Rui and Zhu, Wenye and Li, Xiaofei},
            journal={arXiv preprint arXiv:2204.08765},
            year={2022}
        }
    """
    assert rir.ndim == 1, "rir must be a 1D array."

    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)
    idx_max = int(np.argmax(np.abs(rir)))
    N1 = int(idx_max + time_after_max * sr)
    win = np.empty(shape=rir.shape, dtype=np.float32)
    win[:N1] = 1
    win[N1:] = 10 ** (-q * np.arange(rir.shape[0] - N1))
    rir = rir * win
    return rir, win
