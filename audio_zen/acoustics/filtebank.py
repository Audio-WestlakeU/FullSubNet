import torch


def apply_filterbank(specgram, filterbank):
    """
    Apply a filterbank on specgram

    Args:
        specgram: [..., F, T]
        filterbank: [F, N]. N is the number of filters

    Returns:
        [..., N, T]
    """
    # Pack batch
    shape = specgram.size()
    specgram = specgram.reshape(-1, shape[-2], shape[-1])

    # [C, F, T].T @ [F, M] => [C, T, M].T => [C, M, T]
    mel_specgram = torch.matmul(specgram.transpose(1, 2), filterbank).transpose(1, 2)

    # Unpack batch
    mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

    return mel_specgram


def inverse_filterbank(filtered_specgram, filterbank):
    """
    Left matrix multiply a filterbank.

    References:
        1. https://github.com/timsainb/python_spectrograms_and_inversion
        2. https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.InverseMelScale


    Args:
        filtered_specgram: [..., N ,T]
        filterbank: [F, N]

    Returns:

    """
    # Pack batch
    shape = filtered_specgram.size()
    filtered_specgram = filtered_specgram.reshape(-1, shape[-2], shape[-1])

    specgram = torch.matmul(filterbank, filtered_specgram)

    # Unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram
