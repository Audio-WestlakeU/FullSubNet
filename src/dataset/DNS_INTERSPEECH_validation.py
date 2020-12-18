import os
from pathlib import Path

import librosa

from common.dataset import BaseDataset
from util.acoustic_utils import load_wav


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
        use the absolute path of noisy speeches to find the corresponding clean speech.

        Notes
            1. with_reverb and no_reverb dirs have same-name files. if we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            [waveform...], [waveform...], [reverb|no_reverb] + name
        """
        noisy_file_path = self.noisy_files_list[item]
        speech_parent_dir = Path(noisy_file_path).parents[1].name
        noisy_file_basename = os.path.splitext(os.path.basename(noisy_file_path))[0]
        reverb_remark = ""  # 当语音来自于混响的目录时，会在 noisy filename 前添加 with_reverb

        # speech_type 与 validation 部分要一致，用于区分后续的可视化
        if speech_parent_dir == "with_reverb":
            speech_type = "With_reverb"
        elif speech_parent_dir == "no_reverb":
            speech_type = "No_reverb"
        elif speech_parent_dir == "dns_2_non_english":
            speech_type = "Non_english"
        elif speech_parent_dir == "dns_2_emotion":
            speech_type = "Emotion"
        elif speech_parent_dir == "dns_2_singing":
            speech_type = "Singing"
        else:
            raise NotImplementedError(f"Not supported speech dir: {speech_parent_dir}")

        # 确定带噪语音对应的纯净语音
        file_id = noisy_file_basename.split('_')[-1]
        if speech_parent_dir in ("dns_2_emotion", "dns_2_singing"):
            # synthetic_emotion_1792_snr19_tl-35_fileid_19 => synthetic_emotion_clean_fileid_15
            clean_file_basename = f"synthetic_{speech_type.lower()}_clean_fileid_{file_id}"
        elif speech_parent_dir == "dns_2_non_english":
            # synthetic_german_collection044_14_-04_CFQQgBvv2xQ_snr8_tl-21_fileid_121 => synthetic_clean_fileid_121
            clean_file_basename = f"synthetic_clean_fileid_{file_id}"
        else:
            # clnsp587_Unt_WsHPhfA_snr8_tl-30_fileid_300 => clean_fileid_300
            if speech_parent_dir == "with_reverb":
                reverb_remark = "with_reverb"  # 当语音来自于混响的目录时，会在 noisy filename 前添加 with_reverb
            clean_file_basename = f"clean_fileid_{file_id}"

        clean_file_path = noisy_file_path.replace(f"noisy/{noisy_file_basename}", f"clean/{clean_file_basename}")

        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        clean = load_wav(os.path.abspath(os.path.expanduser(clean_file_path)), sr=self.sr)

        return noisy, clean, reverb_remark + noisy_file_basename, speech_type
