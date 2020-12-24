import argparse
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))
from inspect import getmembers, isfunction
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import util.metrics as metrics
from util.utils import prepare_empty_dir


def load_wav_paths_from_scp(scp_path, to_abs=True):
    wav_paths = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(scp_path)), "r")]
    if to_abs:
        tmp = []
        for path in wav_paths:
            tmp.append(os.path.abspath(os.path.expanduser(path)))
        wav_paths = tmp
    return wav_paths


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def pre_processing(est, ref, specific_dataset):
    ref = Path(ref).expanduser().absolute()
    est = Path(est).expanduser().absolute()

    if ref.is_dir():
        reference_wav_paths = librosa.util.find_files(ref.as_posix())
    else:
        reference_wav_paths = load_wav_paths_from_scp(ref.as_posix())

    if est.is_dir():
        estimated_wav_paths = librosa.util.find_files(est.as_posix())
    else:
        estimated_wav_paths = load_wav_paths_from_scp(est.as_posix())

    if not specific_dataset:
        check_two_aligned_list(reference_wav_paths, estimated_wav_paths)
    else:
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
        est_wav, _ = librosa.load(est_wav_path, sr=sr)
        basename = get_basename(ref_wav_path)

        ref_wav_len = len(ref_wav)
        est_wav_len = len(est_wav)

        if ref_wav_len != est_wav_len:
            # print(f"[Warning] x and y have not the same length, {ref_wav_len} != {est_wav_len}")
            pass

        return basename, metric_function(ref_wav, est_wav[:len(ref_wav)])

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
        description="输入两个目录的 scp 文件，计算各种评价指标的均值",
        epilog="python calculate_metrics.py -E 'est_dir' -R 'ref_dir' -M STOI,PESQ,SI_SDR,V_PESQ"
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
