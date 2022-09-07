import importlib
import os
import time
from typing import Optional

import torch


def load_checkpoint(checkpoint_path, device):
    """Load PyTorch model checkpoint from a given path.

    Args:
        checkpoint_path: path to the checkpoint file. It can be *.pth or *.tar
        device: device to load the checkpoint.

    Returns:
        Model checkpoint.
    """
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of l1 checkpoint."
    model_checkpoint = torch.load(
        os.path.abspath(os.path.expanduser(checkpoint_path)), map_location=device
    )

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # load tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["l1"]


def prepare_empty_dir(dirs, resume=False):
    """If resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert (
                dir_path.exists()
            ), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def check_nan(tensor, key=""):
    if torch.sum(torch.isnan(tensor)) > 0:
        print(f"Found NaN in {key}")


class ExecutionTime:
    """Count execution time.

    Examples:
        timer = ExecutionTime()
        ...
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_module(path: str, args: Optional[dict] = None, initialize: bool = True):
    """Load module or function dynamically with "args".

    Args:
        path: module path in this project.
        args: parameters that will be passed to the Class or the Function in the module.
        initialize: whether to initialize the Class or the Function with args.

    Examples:
        Config items are as follows:

            [model]
            path = "model.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

        This function will:
            1. Load the "model.full_sub_net" module.
            2. Call "FullSubNetModel" Class (or Function) in "model.full_sub_net" module.
            3. If initialize is True:
                instantiate (or call) the Class (or the Function) and pass the parameters (in "[model.args]") to it.
    """
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def print_tensor_info(tensor, flag="Tensor"):
    def floor_tensor(float_tensor):
        return int(float(float_tensor) * 1000) / 1000

    print(
        f"{flag}\n"
        f"\t"
        f"max: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, "
        f"mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}"
    )


def set_requires_grad(nets, requires_grad=False):
    """Set "requies_grad=Fasle" for all the networks to avoid unnecessary computations.

    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def prepare_device(n_gpu: int, keep_reproducibility=False):
    """Choose to use CPU or GPU depend on the value of "n_gpu".

    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.

    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            # ensures that CUDA selects the same convolution algorithm each time
            torch.backends.cudnn.benchmark = False
            # configures PyTorch only to use deterministic implementation
            torch.set_deterministic(True)
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda:0")

    return device


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext
