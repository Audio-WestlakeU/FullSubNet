import os
import random
import sys
from pathlib import Path

import librosa
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "recipes"))
from audio_zen.acoustics.mask import is_clipped, load_wav, activity_detector

candidate_datasets = [
    "~/Datasets/DNS-Challenge-ICASSP/datasets/clean/german_speech/CC_BY_SA_4.0_249hrs_339spk_German_Wikipedia_16k",
    "~/Datasets/DNS-Challenge-ICASSP/datasets/clean/german_speech/M-AILABS_Speech_Dataset",

]  # 候选数据集的目录，支持多个目录
dataset_limit = None
dataset_offset = 0
dist_file = Path("~/Datasets/DNS-Challenge-ICASSP/datasets/german_3s_0.6_30hrs.txt").expanduser().absolute()

# 声学参数
sr = 16000
wav_min_second = 3
activity_threshold = 0.6
total_hrs = 30.0  # 计划收集语音的总时长


def offset_and_limit(data_list, offset, limit):
    data_list = data_list[offset:]
    if limit:
        data_list = data_list[:limit]
    return data_list


if __name__ == '__main__':
    """
    Returns
        scp txt file
    """
    all_wav_path_list = []
    output_wav_path_list = []
    accumulated_time = 0.0

    is_clipped_wav_list = []
    is_low_activity_list = []
    is_too_short_list = []

    for dataset_path in candidate_datasets:
        dataset_path = Path(dataset_path).expanduser().absolute()
        all_wav_path_list += librosa.util.find_files(dataset_path.as_posix(), ext=["wav"])

    all_wav_path_list = offset_and_limit(all_wav_path_list, dataset_offset, dataset_limit)
    random.shuffle(all_wav_path_list)

    for wav_file_path in tqdm(all_wav_path_list, desc="Checking"):
        y = load_wav(wav_file_path, sr=sr)
        wav_duration = len(y) / sr
        wav_file_user_path = wav_file_path.replace(Path(wav_file_path).home().as_posix(), "~")

        is_clipped_wav = is_clipped(y)
        is_low_activity = activity_detector(y) < activity_threshold
        is_too_short = wav_duration < wav_min_second

        if is_too_short:
            is_too_short_list.append(wav_file_user_path)
            continue

        if is_clipped_wav:
            is_clipped_wav_list.append(wav_file_user_path)
            continue

        if is_low_activity:
            is_low_activity_list.append(wav_file_user_path)
            continue

        if (not is_clipped_wav) and (not is_low_activity) and (not is_too_short):
            accumulated_time += wav_duration
            output_wav_path_list.append(wav_file_user_path)

        if accumulated_time >= (total_hrs * 3600):
            break

    with open(dist_file.as_posix(), 'w') as f:
        f.writelines(f"{file_path}\n" for file_path in output_wav_path_list)

    print("=" * 70)
    print("Speech Preprocessing")
    print(f"\t Original files: {len(all_wav_path_list)}")
    print(f"\t Selected files: {accumulated_time / 3600} hrs, {len(output_wav_path_list)} files.")
    print(f"\t is_clipped_wav: {len(is_clipped_wav_list)}")
    print(f"\t is_low_activity: {len(is_low_activity_list)}")
    print(f"\t is_too_short: {len(is_too_short_list)}")
    print(f"\t dist file:")
    print(f"\t {dist_file.as_posix()}")
    print("=" * 70)
