import shutil
import time
from functools import partial
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from joblib import Parallel, delayed
from rich import print
from rich.console import Console
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import audio_zen.metrics as metrics
from audio_zen.acoustics.feature import istft, stft
from audio_zen.acoustics.utils import transform_pesq_range
from audio_zen.utils import ExecutionTime, prepare_empty_dir

plt.switch_backend("agg")
console = Console()


class BaseTrainer:
    def __init__(
        self, dist, rank, config, resume, only_validation, model, loss_function, optimizer
    ):
        self.model = DistributedDataParallel(model.cuda(rank), device_ids=[rank])
        self.optimizer = optimizer
        self.loss_function = loss_function

        # DistributedDataParallel (DDP)
        self.dist = dist
        self.rank = rank

        torch.backends.cudnn.enabled = config["meta"]["cudnn_enable"]
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # Automatic mixed precision (AMP)
        self.use_amp = config["meta"]["use_amp"]
        self.scaler = GradScaler(enabled=self.use_amp)

        # Acoustics
        self.acoustic_config = config["acoustics"]
        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]

        # Supported STFT
        self.torch_stft = partial(
            stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.torch_istft = partial(
            istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_stft = partial(
            librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        self.librosa_istft = partial(
            librosa.istft, hop_length=hop_length, win_length=win_length
        )

        # Trainer.train in the config
        self.train_config = config["trainer"]["train"]
        self.epochs = self.train_config["epochs"]
        self.save_checkpoint_interval = self.train_config["save_checkpoint_interval"]
        self.clip_grad_norm_value = self.train_config["clip_grad_norm_value"]
        assert (
            self.save_checkpoint_interval >= 1
        ), "Check the 'save_checkpoint_interval' parameter in the config. It should be large than one."

        # Trainer.validation in the config
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["validation_interval"]
        self.save_max_metric_score = self.validation_config["save_max_metric_score"]
        assert (
            self.validation_interval >= 1
        ), "Check the 'validation_interval' parameter in the config. It should be large than one."

        # Trainer.visualization in the config
        self.visualization_config = config["trainer"]["visualization"]

        # In the 'train.py' file, if the 'resume' item is 'True', we will update the following args:
        self.start_epoch = 1
        self.best_score = -np.inf if self.save_max_metric_score else np.inf
        self.save_dir = (
            Path(config["meta"]["save_dir"]).expanduser().absolute()
            / config["meta"]["experiment_name"]
        )
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        self.source_code_dir = Path(__file__).expanduser().absolute().parent.parent.parent

        if resume:
            self._resume_checkpoint()

        # Debug validation, which skips training
        self.only_validation = only_validation

        if config["meta"]["preloaded_model_path"]:
            self._preload_model(Path(config["preloaded_model_path"]))

        if self.rank == 0:
            prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

            self.writer = SummaryWriter(
                self.logs_dir.as_posix(), max_queue=5, flush_secs=30
            )
            self.writer.add_text(
                tag="Configuration",
                text_string=f"<pre>  \n{toml.dumps(config)}  \n</pre>",
                global_step=1,
            )

            print("The configurations are as follows: ")
            print(config)  # except "\n"

            # Backup of config
            with open(
                (self.save_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.toml").as_posix(),
                "w",
            ) as handle:
                toml.dump(config, handle)

            # Backup of project code
            shutil.copytree(
                src=self.source_code_dir.as_posix(),
                dst=(self.save_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}").as_posix(),
            )

            self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        model_path = model_path.expanduser().absolute()
        assert (
            model_path.exists()
        ), f"The file {model_path.as_posix()} is not exist. please check path."

        model_checkpoint = torch.load(model_path.as_posix(), map_location="cpu")
        self.model.load_state_dict(model_checkpoint["model"], strict=False)
        self.model.to(self.rank)

        if self.rank == 0:
            print(f"Model preloaded successfully from {model_path.as_posix()}.")

    def _resume_checkpoint(self):
        """
        Resume the experiment from the latest checkpoint.
        """
        latest_model_path = (
            self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        )
        assert (
            latest_model_path.exists()
        ), f"{latest_model_path} does not exist, can not load latest checkpoint."

        # Load it on the CPU and later use .to(device) on the model
        # Maybe slightly slow than use map_location="cuda:<...>"
        # https://stackoverflow.com/questions/61642619/pytorch-distributed-data-parallel-confusion
        checkpoint = torch.load(latest_model_path.as_posix(), map_location="cpu")

        # Make sure all processes (GPUs) do not start loading before the saving is finished.
        # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
        self.dist.barrier()

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        # self.model.to(self.rank)

        if self.rank == 0:
            print(
                f"Model checkpoint is loaded. Training will begin at epoch {self.start_epoch}."
            )

    def _save_checkpoint(self, epoch, is_best_epoch=False):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - epoch
            - best metric score in historical epochs
            - optimizer parameters
            - model parameters

        Args:
            is_best_epoch (bool): In the current epoch, if the model get a best metric score (is_best_epoch=True),
                                the checkpoint of model will be saved as "<save_dir>/checkpoints/best_model.tar".
        """
        print(f"\t Saving the model checkpoint of epoch {epoch}...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        # Saved in "latest_model.tar"
        # Contains all checkpoint information, including the optimizer parameters, the model parameters, etc.
        # New checkpoint will overwrite the older one.
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())

        # "model_{epoch_number}.pth"
        # Contains only model.
        torch.save(
            state_dict["model"],
            (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix(),
        )

        # If the model get a best metric score (means "is_best_epoch=True") in the current epoch,
        # the model checkpoint will be saved as "best_model.tar"
        # The newer best-scored checkpoint will overwrite the older one.
        if is_best_epoch:
            print(f"\t :smiley: Found a best score in the epoch {epoch}, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

    def _is_best_epoch(self, score, save_max_metric_score=True):
        """
        Check if the current model got the best metric score
        """
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(models: list):
        print(
            f"This project contains {len(models)} models, the number of the parameters is: "
        )

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\Model {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(
            f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
        )

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def spec_audio_visualization(self, noisy, enhanced, clean, name, epoch, mark=""):
        self.writer.add_audio(
            f"{mark}_Speech/{name}_Noisy", noisy, epoch, sample_rate=16000
        )
        self.writer.add_audio(
            f"{mark}_Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000
        )
        self.writer.add_audio(
            f"{mark}_Speech/{name}_Clean", clean, epoch, sample_rate=16000
        )

        # Visualize the spectrogram of noisy speech, clean speech, and enhanced speech
        noisy_mag, _ = librosa.magphase(
            self.librosa_stft(noisy, n_fft=320, hop_length=160, win_length=320)
        )
        enhanced_mag, _ = librosa.magphase(
            self.librosa_stft(enhanced, n_fft=320, hop_length=160, win_length=320)
        )
        clean_mag, _ = librosa.magphase(
            self.librosa_stft(clean, n_fft=320, hop_length=160, win_length=320)
        )
        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([noisy_mag, enhanced_mag, clean_mag]):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(
                librosa.amplitude_to_db(mag),
                cmap="magma",
                y_axis="linear",
                ax=axes[k],
                sr=16000,
            )
        plt.tight_layout()
        self.writer.add_figure(f"{mark}_Spectrogram/{name}", fig, epoch)

    def metrics_visualization(
        self,
        noisy_list,
        clean_list,
        enhanced_list,
        metrics_list,
        epoch,
        num_workers=10,
        mark="",
    ):
        """Get metrics on validation dataset by paralleling.

        Notes:
            1. You can register other metrics, but STOI and WB_PESQ metrics must be existence. These two metrics are
             used for checking if the current epoch is a "best epoch."
            2. If you want to use a new metric, you must register it in "utile.
        """
        assert (
            "STOI" in metrics_list and "WB_PESQ" in metrics_list
        ), "'STOI' and 'WB_PESQ' must be exist."

        # Check if the metric is registered in "util.metrics" file.
        for i in metrics_list:
            assert (
                i in metrics.REGISTERED_METRICS.keys()
            ), f"{i} is not registered, please check 'util.metrics' file."

        stoi_mean = 0.0
        wb_pesq_mean = 0.0
        for metric_name in metrics_list:
            score_on_noisy = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est)
                for ref, est in zip(clean_list, noisy_list)
            )
            score_on_enhanced = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est)
                for ref, est in zip(clean_list, enhanced_list)
            )

            # Add mean value of the metric to tensorboard
            mean_score_on_noisy = np.mean(score_on_noisy)
            mean_score_on_enhanced = np.mean(score_on_enhanced)
            self.writer.add_scalars(
                f"{mark}_Validation/{metric_name}",
                {"Noisy": mean_score_on_noisy, "Enhanced": mean_score_on_enhanced},
                epoch,
            )

            if metric_name == "STOI":
                stoi_mean = mean_score_on_enhanced

            if metric_name == "WB_PESQ":
                wb_pesq_mean = transform_pesq_range(mean_score_on_enhanced)

        return (stoi_mean + wb_pesq_mean) / 2

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.rank == 0:
                print(f"{'=' * 15} epoch {epoch} {'=' * 15}")
                print("[0 seconds] Begin training...")

            # [debug validation] Only run validation (only use the first GPU (process))
            # inference + calculating metrics + saving checkpoints
            if self.only_validation and self.rank == 0:
                self._set_models_to_eval_mode()
                metric_score = self._validation_epoch(epoch)

                if self._is_best_epoch(
                    metric_score, save_max_metric_score=self.save_max_metric_score
                ):
                    self._save_checkpoint(epoch, is_best_epoch=True)

                # Skip the following regular training, saving checkpoints, and validation
                continue

            # Regular training
            timer = ExecutionTime()
            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            #  Regular save checkpoints
            if (
                self.rank == 0
                and self.save_checkpoint_interval != 0
                and (epoch % self.save_checkpoint_interval == 0)
            ):
                self._save_checkpoint(epoch)

            # Regular validation
            if self.rank == 0 and (epoch % self.validation_interval == 0):
                print(
                    f"[{timer.duration()} seconds] Training is finished, and validation is in progress..."
                )

                self._set_models_to_eval_mode()
                metric_score = self._validation_epoch(epoch)

                if self._is_best_epoch(
                    metric_score, save_max_metric_score=self.save_max_metric_score
                ):
                    self._save_checkpoint(epoch, is_best_epoch=True)

            if self.rank == 0:
                print(f"[{timer.duration()} seconds] This epoch is finished.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError
