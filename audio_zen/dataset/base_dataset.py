from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _offset_and_limit(dataset_list, offset, limit):
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        return dataset_list

    @staticmethod
    def _parse_snr_range(snr_range):
        assert (
            len(snr_range) == 2
        ), f"The range of SNR should be [low, high], not {snr_range}."
        assert (
            snr_range[0] <= snr_range[-1]
        ), f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list
