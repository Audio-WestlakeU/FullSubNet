import os
from pathlib import Path

import librosa

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import load_wav
from audio_zen.utils import basename


class Dataset(BaseDataset):
    def __init__(
            self,
            dataset_dir_list,
            sr,
    ):
        """
        Construct DNS validation set

        synthetic/
            with_reverb/
                noisy/
                clean_y/
            no_reverb/
                noisy/
                clean_y/
        """
        super(Dataset, self).__init__()
        noisy_files_list = []

        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_files_list += librosa.util.find_files((dataset_dir / "noisy").as_posix())

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of the noisy speech to find the corresponding clean speech.

        Notes
            with_reverb and no_reverb dirs have same-named files.
            If we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            noisy: [waveform...], clean: [waveform...], type: [reverb|no_reverb] + name
        """
        noisy_file_path = self.noisy_files_list[item]
        parent_dir = Path(noisy_file_path).parents[1].name
        noisy_filename, _ = basename(noisy_file_path)

        reverb_remark = ""  # When the speech comes from reverb_dir, insert "with_reverb" before the filename

        # speech_type 与 validation 部分要一致，用于区分后续的可视化
        if parent_dir == "with_reverb":
            speech_type = "With_reverb"
        elif parent_dir == "no_reverb":
            speech_type = "No_reverb"
        elif parent_dir == "dns_2_non_english":
            speech_type = "Non_english"
        elif parent_dir == "dns_2_emotion":
            speech_type = "Emotion"
        elif parent_dir == "dns_2_singing":
            speech_type = "Singing"
        else:
            raise NotImplementedError(f"Not supported dir: {parent_dir}")

        # Find the corresponding clean speech using "parent_dir" and "file_id"
        file_id = noisy_filename.split("_")[-1]
        if parent_dir in ("dns_2_emotion", "dns_2_singing"):
            # e.g., synthetic_emotion_1792_snr19_tl-35_fileid_19 => synthetic_emotion_clean_fileid_15
            clean_filename = f"synthetic_{speech_type.lower()}_clean_fileid_{file_id}"
        elif parent_dir == "dns_2_non_english":
            # e.g., synthetic_german_collection044_14_-04_CFQQgBvv2xQ_snr8_tl-21_fileid_121 => synthetic_clean_fileid_121
            clean_filename = f"synthetic_clean_fileid_{file_id}"
        else:
            # e.g., clnsp587_Unt_WsHPhfA_snr8_tl-30_fileid_300 => clean_fileid_300
            if parent_dir == "with_reverb":
                reverb_remark = "with_reverb"
            clean_filename = f"clean_fileid_{file_id}"

        clean_file_path = noisy_file_path.replace(f"noisy/{noisy_filename}", f"clean/{clean_filename}")

        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        clean = load_wav(os.path.abspath(os.path.expanduser(clean_file_path)), sr=self.sr)

        return noisy, clean, reverb_remark + noisy_filename, speech_type
