from pathlib import Path

import librosa
import numpy as np

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import basename


class Dataset(BaseDataset):
    def __init__(self,
                 dataset_dir_list,
                 sr,
                 ):
        """
        Args:
            noisy_dataset_dir_list (str or list): noisy dir or noisy dir list
        """
        super().__init__()
        assert isinstance(dataset_dir_list, list)
        self.sr = sr

        noisy_file_path_list = []
        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_file_path_list += librosa.util.find_files(dataset_dir.as_posix())  # Sorted

        self.noisy_file_path_list = noisy_file_path_list
        self.length = len(self.noisy_file_path_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_file_path = self.noisy_file_path_list[item]
        noisy_y = librosa.load(noisy_file_path, sr=self.sr)[0]
        noisy_y = noisy_y.astype(np.float32)

        return noisy_y, basename(noisy_file_path)[0]
