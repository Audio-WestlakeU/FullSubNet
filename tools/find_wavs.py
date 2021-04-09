from pathlib import Path

import librosa

####
# Parameters
dir_list = [
    "~/Datasets/keyboard1_shrink/0/noisy",
    "~/Datasets/keyboard1_shrink/1/noisy",
    "~/Datasets/keyboard1_shrink/2/noisy",
    "~/Datasets/keyboard1_shrink/3/noisy",
    "~/Datasets/keyboard1_shrink/4/noisy",
    "~/Datasets/keyboard1_shrink/5/noisy",
    "~/Datasets/keyboard1_shrink/6/noisy",
    "~/Datasets/keyboard1_shrink/7/noisy",
    "~/Datasets/keyboard1_shrink/8/noisy",
    "~/Datasets/keyboard1_shrink/9/noisy",
]
dist_path = Path("~/Datasets/hongh4_train.txt").expanduser().absolute()
#####


def main():
    file_path_list = []
    for dataset_dir in dir_list:
        dataset_dir = Path(dataset_dir).expanduser().absolute()
        file_path_list += librosa.util.find_files(dataset_dir.as_posix())  # Sorted

    print(f"Length: {len(file_path_list)}")

    # filter
    tmp = []
    for i, line in enumerate(file_path_list):
        tmp.append(
            f"spk1___{i}___utt1___90___0_300	{line}\n"
        )

    with open(dist_path.as_posix(), "w") as f:
        f.writelines(tmp)


main()
