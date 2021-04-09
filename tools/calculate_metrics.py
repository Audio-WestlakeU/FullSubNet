import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
print(sys.path)

from inspect import getmembers, isfunction
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import audio_zen.metrics as metrics
from audio_zen.utils import prepare_empty_dir


def load_wav_paths_from_scp(scp_path, to_abs=True):
    wav_paths = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(scp_path)), "r")]
    if to_abs:
        tmp = []
        for path in wav_paths:
            tmp.append(os.path.abspath(os.path.expanduser(path)))
        wav_paths = tmp
    return wav_paths


def shrink_multi_channel_path(
        full_dataset_list: list,
        num_channels: int
) -> list:
    """

    Args:
        full_dataset_list: [
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic1.wav
            ...
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic2.wav
        ]
        num_channels:

    Returns:

    """
    assert len(full_dataset_list) % num_channels == 0, "Num error"

    shrunk_dataset_list = []
    for index in range(0, len(full_dataset_list), num_channels):
        full_path = full_dataset_list[index]
        shrunk_path = f"{'_'.join(full_path.split('_')[:-1])}.wav"
        shrunk_dataset_list.append(shrunk_path)

    assert len(shrunk_dataset_list) == len(full_dataset_list) // num_channels
    return shrunk_dataset_list


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def pre_processing(est, ref, specific_dataset=None):
    ref = Path(ref).expanduser().absolute()
    est = Path(est).expanduser().absolute()

    if ref.is_dir():
        reference_wav_paths = librosa.util.find_files(ref.as_posix(), ext="wav")
    else:
        reference_wav_paths = load_wav_paths_from_scp(ref.as_posix())

    if est.is_dir():
        estimated_wav_paths = librosa.util.find_files(est.as_posix(), ext="wav")
    else:
        estimated_wav_paths = load_wav_paths_from_scp(est.as_posix())

    if not specific_dataset:
        # 默认情况下，两个列表应该是一一对应的
        check_two_aligned_list(reference_wav_paths, estimated_wav_paths)
    else:
        # 针对不同的数据集，进行手工对齐，保证两个列表一一对应
        reordered_estimated_wav_paths = []
        if specific_dataset == "dns_1":
            # 按照 reference_wav_paths 中文件的后缀名重排 estimated_wav_paths
            # 提取后缀
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    est_basename = get_basename(est_path)
                    if "clean_" + "_".join(est_basename.split("_")[-2:]) == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif specific_dataset == "dns_2":
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    # synthetic_french_acejour_orleans_sb_64kb-01_jbq2HJt9QXw_snr14_tl-26_fileid_47
                    # synthetic_clean_fileid_47
                    est_basename = get_basename(est_path)
                    file_id = est_basename.split('_')[-1]
                    if f"synthetic_clean_fileid_{file_id}" == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif specific_dataset == "maxhub_noisy":
            # Reference_channel = 0
            # 寻找对应的干净语音
            reference_channel = 0
            print(f"Found #files: {len(reference_wav_paths)}")
            for est_path in estimated_wav_paths:
                # MC0604W0154_room4_rev_RT600.1_mic1_micpos1.5p0.5p1.84_srcpos4.507p1.5945p1.3_langle180_angle20_ds3.2_kesou_kesou_mic1.wav
                est_basename = get_basename(est_path)  # 带噪的
                for ref_path in reference_wav_paths:
                    ref_basename = get_basename(ref_path)

        else:
            raise NotImplementedError(f"Not supported specific dataset {specific_dataset}.")
        estimated_wav_paths = reordered_estimated_wav_paths

    return reference_wav_paths, estimated_wav_paths


def check_two_aligned_list(a, b):
    assert len(a) == len(b), "两个列表中的长度不等."
    for z, (i, j) in enumerate(zip(a, b), start=1):
        assert get_basename(i) == get_basename(j), f"两个列表中存在不相同的文件名，行数为: {z}" \
                                                   f"\n\t {i}" \
                                                   f"\n\t{j}"


def compute_metric(reference_wav_paths, estimated_wav_paths, sr, metric_type="SI_SDR"):
    metrics_dict = {o[0]: o[1] for o in getmembers(metrics) if isfunction(o[1])}
    assert metric_type in metrics_dict, f"不支持的评价指标： {metric_type}"
    metric_function = metrics_dict[metric_type]

    def calculate_metric(ref_wav_path, est_wav_path):
        ref_wav, _ = librosa.load(ref_wav_path, sr=sr)
        est_wav, _ = librosa.load(est_wav_path, sr=sr, mono=False)
        if est_wav.ndim > 1:
            est_wav = est_wav[0]

        basename = get_basename(ref_wav_path)

        ref_wav_len = len(ref_wav)
        est_wav_len = len(est_wav)

        if ref_wav_len != est_wav_len:
            print(f"[Warning] ref {ref_wav_len} and est {est_wav_len} are not in the same length")
            pass

        return basename, metric_function(ref_wav[:len(est_wav)], est_wav)

    metrics_result_store = Parallel(n_jobs=40)(
        delayed(calculate_metric)(ref, est) for ref, est in tqdm(zip(reference_wav_paths, estimated_wav_paths))
    )
    return metrics_result_store


def main(args):
    sr = args.sr
    metric_types = args.metric_types
    export_dir = args.export_dir
    specific_dataset = args.specific_dataset.lower()

    # 通过指定的 scp 文件或目录获取全部的 wav 样本
    reference_wav_paths, estimated_wav_paths = pre_processing(args.estimated, args.reference, specific_dataset)

    if export_dir:
        export_dir = Path(export_dir).expanduser().absolute()
        prepare_empty_dir([export_dir])

    print(f"=== {args.estimated} === {args.reference} ===")
    for metric_type in metric_types.split(","):
        metrics_result_store = compute_metric(reference_wav_paths, estimated_wav_paths, sr, metric_type=metric_type)

        # Print result
        metric_value = np.mean(list(zip(*metrics_result_store))[1])
        print(f"{metric_type}: {metric_value}")

        # Export result
        if export_dir:
            import tablib

            export_path = export_dir / f"{metric_type}.xlsx"
            print(f"Export result to {export_path}")

            headers = ("Speech", f"{metric_type}")
            metric_seq = [[basename, metric_value] for basename, metric_value in metrics_result_store]
            data = tablib.Dataset(*metric_seq, headers=headers)
            with open(export_path.as_posix(), "wb") as f:
                f.write(data.export("xlsx"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="输入两个目录或列表，计算各种评价指标的均值",
        epilog="python calculate_metrics.py -E 'est_dir' -R 'ref_dir' -M SI_SDR,STOI,WB_PESQ,NB_PESQ,SSNR,LSD,SRMR"
    )
    parser.add_argument("-R", "--reference", required=True, type=str, help="")
    parser.add_argument("-E", "--estimated", required=True, type=str, help="")
    parser.add_argument("-M", "--metric_types", required=True, type=str, help="哪个评价指标，要与 util.metrics 中的内容一致.")
    parser.add_argument("--sr", type=int, default=16000, help="采样率")
    parser.add_argument("-D", "--export_dir", type=str, default="", help="")
    parser.add_argument("--limit", type=int, default=None, help="[正在开发]从列表中读取文件的上限数量.")
    parser.add_argument("--offset", type=int, default=0, help="[正在开发]从列表中指定位置开始读取文件.")
    parser.add_argument("-S", "--specific_dataset", type=str, default="", help="指定数据集类型，e.g. DNS_1, DNS_2, 大小写均可")
    args = parser.parse_args()
    main(args)

    """
    TODO
    1. 语音为多通道时如何算
    2. 支持 register, 默认情况下应该计算所有 register 中语音
    """
