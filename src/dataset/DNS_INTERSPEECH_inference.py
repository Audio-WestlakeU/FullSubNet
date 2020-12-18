import os

import librosa
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, noisy_dataset, limit, offset, sr):
        """
        Args:
            noisy_dataset (str): noisy dir (wav format files) or noisy filenames list
        """
        noisy_dataset = os.path.abspath(os.path.expanduser(noisy_dataset))

        if os.path.isfile(noisy_dataset):
            noisy_wav_files = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(noisy_dataset)), "r")]
            if offset:
                noisy_wav_files = noisy_wav_files[offset:]
            if limit:
                noisy_wav_files = noisy_wav_files[:limit]
        elif os.path.isdir(noisy_dataset):
            noisy_wav_files = librosa.util.find_files(noisy_dataset, ext="wav", limit=limit if limit else None, offset=offset)
        else:
            raise FileNotFoundError(f"Please Check {noisy_dataset}")

        print(f"Num of noisy files in {noisy_dataset}: {len(noisy_wav_files)}")

        self.length = len(noisy_wav_files)
        self.noisy_wav_files = noisy_wav_files
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path = self.noisy_wav_files[item]
        basename = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy = librosa.load(noisy_path, sr=self.sr)[0]

        return noisy, basename
