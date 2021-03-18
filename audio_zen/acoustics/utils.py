import librosa


def transform_pesq_range(pesq_score):
    """
    transform PESQ metric range from [-0.5 ~ 4.5] to [0 ~ 1]
    """
    return (pesq_score + 0.5) / 5


def load_wav(path, sr=16000):
    return librosa.load(path, sr=sr)[0]