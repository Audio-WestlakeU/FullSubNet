import time
from functools import partial
from pathlib import Path

import colorful
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from joblib import Parallel, delayed
from torch.cuda.amp import GradScaler

import util.metrics as metrics
from util import visualization
from util.acoustic_utils import stft, istft, transform_pesq_range
from util.utils import prepare_empty_dir, ExecutionTime

plt.switch_backend('agg')


class BaseTrainer:
    def __init__(self, dist, rank, config, resume: bool, model, loss_function, optimizer):
        self.color_tool = colorful
        self.color_tool.use_style("solarized")

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # DistributedDataParallel (DDP)
        self.rank = rank
        self.dist = dist

        # Automatic mixed precision (AMP)
        self.use_amp = config["meta"]["use_amp"]
        self.scaler = GradScaler(enabled=self.use_amp)

        # Acoustics
        self.acoustic_config = config["acoustic"]

        # Supported STFT
        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]
        self.torch_stft = partial(stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.rank)
        self.istft = partial(istft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, device=self.rank)
        self.librosa_stft = partial(librosa.stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.librosa_istft = partial(librosa.istft, hop_length=hop_length, win_length=win_length)

        # Trainer.train in config
        self.train_config = config["trainer"]["train"]
        self.epochs = self.train_config["epochs"]
        self.save_checkpoint_interval = self.train_config["save_checkpoint_interval"]
        self.clip_grad_norm_value = self.train_config["clip_grad_norm_value"]
        assert self.save_checkpoint_interval >= 1

        # Trainer.validation in config
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["validation_interval"]
        self.save_max_metric_score = self.validation_config["save_max_metric_score"]
        assert self.validation_interval >= 1

        # Trainer.visualization in config
        self.visualization_config = config["trainer"]["visualization"]

        # In the 'train.py' file, if the 'resume' item is True, we will update the following args:
        self.start_epoch = 1
        self.best_score = -np.inf if self.save_max_metric_score else np.inf
        self.save_dir = Path(config["meta"]["save_dir"]).expanduser().absolute() / config["meta"]["experiment_name"]
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"

        if resume:
            self._resume_checkpoint()

        if config["meta"]["preloaded_model_path"]:
            self._preload_model(Path(config["preloaded_model_path"]))

        if self.rank == 0:
            prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

            self.writer = visualization.writer(self.logs_dir.as_posix())
            self.writer.add_text(
                tag="Configuration",
                text_string=f"<pre>  \n{toml.dumps(config)}  \n</pre>",
                global_step=1
            )

            print(self.color_tool.cyan("The configurations are as follows: "))
            print(self.color_tool.cyan("=" * 40))
            print(self.color_tool.cyan(toml.dumps(config)[:-1]))  # except "\n"
            print(self.color_tool.cyan("=" * 40))

            with open((self.save_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
                toml.dump(config, handle)

            self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        model_path = model_path.expanduser().absolute()
        assert model_path.exists(), f"The file {model_path.as_posix()} is not exist. please check path."

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        model_checkpoint = torch.load(model_path.as_posix(), map_location=map_location)
        self.model.load_state_dict(model_checkpoint["model"], strict=False)

        if self.rank == 0:
            print(f"Model preloaded successfully from {model_path.as_posix()}.")

    def _resume_checkpoint(self):
        """
        Resume experiment from the latest checkpoint.
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(latest_model_path.as_posix(), map_location=map_location)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.model.load_state_dict(checkpoint["model"])

        if self.rank == 0:
            print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best_epoch=False):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters

        Args:
            is_best_epoch (bool): In current epoch, if the model get a best metric score (is_best_epoch=True),
                                the checkpoint of model will be saved as "<save_dir>/checkpoints/best_model.tar".
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # TODO
        # 统一训练与推理时的处理方式："module.*"...
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "model": self.model.state_dict()
        }

        # "latest_model.tar"
        # Contains all checkpoint information, including the optimizer parameters, the model parameters, etc.
        # New checkpoint will overwrite the older one.
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())

        # "model_{epoch_number}.tar"
        # Contains all checkpoint information, like "latest_model.tar". However, the newer information will no overwrite the older one.
        torch.save(state_dict, (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.tar").as_posix())

        # If the model get a best metric score (is_best_epoch=True) in the current epoch,
        # the model checkpoint will be saved as "best_model.tar."
        # The newer best-scored checkpoint will overwrite the older one.
        if is_best_epoch:
            print(self.color_tool.red(f"\t Found a best score in the {epoch} epoch, saving..."))
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
        print(f"This project contains {len(models)} models, the number of the parameters is: ")

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def spec_audio_visualization(self, noisy, enhanced, clean, name, epoch, mark=""):
        self.writer.add_audio(f"{mark}_Speech/{name}_Noisy", noisy, epoch, sample_rate=16000)
        self.writer.add_audio(f"{mark}_Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
        self.writer.add_audio(f"{mark}_Speech/{name}_Clean", clean, epoch, sample_rate=16000)

        # Visualize the spectrogram of noisy speech, clean speech, and enhanced speech
        noisy_mag, _ = librosa.magphase(self.librosa_stft(noisy, n_fft=320, hop_length=160, win_length=320))
        enhanced_mag, _ = librosa.magphase(self.librosa_stft(enhanced, n_fft=320, hop_length=160, win_length=320))
        clean_mag, _ = librosa.magphase(self.librosa_stft(clean, n_fft=320, hop_length=160, win_length=320))
        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([noisy_mag, enhanced_mag, clean_mag]):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
        plt.tight_layout()
        self.writer.add_figure(f"{mark}_Spectrogram/{name}", fig, epoch)

    def metrics_visualization(self, noisy_list, clean_list, enhanced_list, metrics_list, epoch, num_workers=10, mark=""):
        """
        Get metrics on validation dataset by paralleling.

        Notes:
            1. You can register other metrics, but STOI and WB_PESQ metrics must be existence. These two metrics are
             used for checking if the current epoch is a "best epoch."
            2. If you want to use a new metric, you must register it in "util.metrics" file.
        """
        assert "STOI" in metrics_list and "WB_PESQ" in metrics_list, "'STOI' and 'WB_PESQ' must be existence."

        # Check if the metric is registered in "util.metrics" file.
        for i in metrics_list:
            assert i in metrics.REGISTERED_METRICS.keys(), f"{i} is not registered, please check 'util.metrics' file."

        stoi_mean = 0.0
        wb_pesq_mean = 0.0
        for metric_name in metrics_list:
            score_on_noisy = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est) for ref, est in zip(clean_list, noisy_list)
            )
            score_on_enhanced = Parallel(n_jobs=num_workers)(
                delayed(metrics.REGISTERED_METRICS[metric_name])(ref, est) for ref, est in zip(clean_list, enhanced_list)
            )

            # Add the mean value of the metric to tensorboard
            mean_score_on_noisy = np.mean(score_on_noisy)
            mean_score_on_enhanced = np.mean(score_on_enhanced)
            self.writer.add_scalars(f"{mark}_Validation/{metric_name}", {
                "Noisy": mean_score_on_noisy,
                "Enhanced": mean_score_on_enhanced
            }, epoch)

            if metric_name == "STOI":
                stoi_mean = mean_score_on_enhanced

            if metric_name == "WB_PESQ":
                wb_pesq_mean = transform_pesq_range(mean_score_on_enhanced)

        return (stoi_mean + wb_pesq_mean) / 2

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.rank == 0:
                print(self.color_tool.yellow(f"{'=' * 15} {epoch} epoch {'=' * 15}"))
                print("[0 seconds] Begin training...")

            timer = ExecutionTime()
            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            # Only use the first GPU (process) to the validation.
            if self.rank == 0:
                if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                    self._save_checkpoint(epoch)

                if epoch % self.validation_interval == 0:
                    print(f"[{timer.duration()} seconds] Training has finished, validation is in progress...")

                    self._set_models_to_eval_mode()
                    metric_score = self._validation_epoch(epoch)

                    if self._is_best_epoch(metric_score, save_max_metric_score=self.save_max_metric_score):
                        self._save_checkpoint(epoch, is_best_epoch=True)

                    print(f"[{timer.duration()} seconds] This epoch has finished.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError
