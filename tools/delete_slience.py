import os
from pathlib import Path

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

###
noisy_dir = Path("~/Datasets/simulation_array26cm_20210119_shuf100/noisy").expanduser().absolute()
clean_dir = Path("~/Datasets/simulation_array26cm_20210119_shuf100/clean").expanduser().absolute()
text_dir = Path("~/Datasets/simulation_array26cm_20210119_shuf100/txt").expanduser().absolute()
dist_dir = Path("~/Datasets/simulation_array26cm_20210119_shuf100/dist_single").expanduser().absolute()
(dist_dir / "noisy").mkdir(exist_ok=True, parents=True)
(dist_dir / "clean").mkdir(exist_ok=True)
####

noisy_file_paths = librosa.util.find_files(noisy_dir.as_posix(), ext="wav")

for noisy_file_path in tqdm(noisy_file_paths):
    basename = os.path.basename(noisy_file_path)
    mark = os.path.splitext(os.path.basename(noisy_file_path))[0].split("_")[0:2]
    mark = "_".join(mark)  # single_AF0976
    print(mark)
    if mark[:6] != "single":
        continue

    clean_file_path = clean_dir / basename
    txt_file_path = text_dir / (mark + ".wav.txt")

    noisy_wav, _ = librosa.load(noisy_file_path, sr=16000, mono=False)
    clean_wav, _ = librosa.load(clean_file_path, sr=16000, mono=False)

    valid_noisy_wav = np.array([])
    valid_clean_wav = np.array([])
    with open(txt_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        name, start_time, end_time = line.split(" ")
        if name != "sil":
            if valid_clean_wav.size == 0:
                valid_noisy_wav = noisy_wav[:, int(start_time):int(end_time)]
                valid_clean_wav = clean_wav[int(start_time):int(end_time)]
            else:
                valid_noisy_wav = np.concatenate((valid_noisy_wav, noisy_wav[:, int(start_time):int(end_time)]), axis=-1)
                valid_clean_wav = np.concatenate((valid_clean_wav, clean_wav[int(start_time):int(end_time)]))

        soundfile.write((dist_dir / "noisy" / basename).as_posix(), valid_noisy_wav.T, samplerate=16000)
        soundfile.write((dist_dir / "clean" / basename).as_posix(), valid_clean_wav, samplerate=16000)
