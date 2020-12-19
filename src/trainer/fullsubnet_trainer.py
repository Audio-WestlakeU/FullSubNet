import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from common.trainer import BaseTrainer
from util.acoustic_utils import mag_phase, get_complex_ideal_ratio_mask, drop_sub_band

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            dist,
            rank,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(dist, rank, config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy, clean in tqdm(self.train_dataloader, desc=f"Training {self.rank}"):
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
            ground_truth_cIRM = drop_sub_band(
                ground_truth_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_sub_batches
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                pred_cRM = self.model(noisy_mag)
                pred_cRM = pred_cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(ground_truth_cIRM, pred_cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        if self.rank == 0:
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        for i, (noisy, clean, name, speech_type) in tqdm(enumerate(self.valid_dataloader), desc="Validation"):
            assert len(name) == 1, "The batch size of validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_complex = self.torch_stft(noisy)
            clean_complex = self.torch_stft(clean)

            noisy_mag, _ = mag_phase(noisy_complex)
            ground_truth_cIRM = get_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]

            noisy_mag = noisy_mag.unsqueeze(1)
            pred_cRM = self.model(noisy_mag)
            pred_cRM = pred_cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(ground_truth_cIRM, pred_cRM)

            lim = 9.9
            pred_cRM = lim * (pred_cRM >= lim) - lim * (pred_cRM <= -lim) + pred_cRM * (torch.abs(pred_cRM) < lim)
            pred_cRM = -10 * torch.log((10 - pred_cRM) / (10 + pred_cRM))

            enhanced_real = pred_cRM[..., 0] * noisy_complex[..., 0] - pred_cRM[..., 1] * noisy_complex[..., 1]
            enhanced_imag = pred_cRM[..., 1] * noisy_complex[..., 0] + pred_cRM[..., 0] * noisy_complex[..., 1]
            enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)

            enhanced = self.istft(enhanced_complex, length=noisy.size(-1), use_mag_phase=False)

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            """=== === === Visualization === === ==="""
            # Separated Loss
            loss_list[speech_type] += loss
            item_idx_list[speech_type] += 1

            if item_idx_list[speech_type] <= visualization_n_samples:
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]
