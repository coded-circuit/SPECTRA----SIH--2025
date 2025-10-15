from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from isro_pkg.data.datasets import make_dataloaders, make_file_list
from isro_pkg.models.gdnet import CrossModalGDNet
from isro_pkg.utils.metrics import compute_metrics
from isro_pkg.utils.viz import visualize_validation_samples_enhanced


@dataclass
class TrainConfig:
    data_root: str
    subset_size: Optional[int] = 30000
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-5
    base_channels: int = 128
    scale_factor: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def validate(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    ssim_accum = 0.0
    psnr_accum = 0.0
    loss_accum = 0.0
    total_samples = 0
    criterion = nn.L1Loss()
    saved = {"inputs": [], "targets": [], "outputs": []}
    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            thermal_in = inputs[:, :2, :, :]
            rgb_in = inputs[:, 2:, :, :]
            outputs = model(thermal_in, rgb_in)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            loss_accum += loss.item() * batch_size
            batch_ssim, batch_psnr = compute_metrics(outputs, targets)
            ssim_accum += batch_ssim * batch_size
            psnr_accum += batch_psnr * batch_size
            total_samples += batch_size
            if batch_idx == 0:
                saved["inputs"].append(inputs.cpu())
                saved["targets"].append(targets.cpu())
                saved["outputs"].append(outputs.cpu())
            val_loader_tqdm.set_postfix(
                {"Loss": f"{loss.item():.4f}", "SSIM": f"{batch_ssim:.4f}", "PSNR": f"{batch_psnr:.2f}"})
    avg_loss = loss_accum / total_samples
    avg_ssim = ssim_accum / total_samples
    avg_psnr = psnr_accum / total_samples
    print(
        f"Validation - Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")
    model.train()
    inputs_sample = torch.cat(saved["inputs"], dim=0)
    targets_sample = torch.cat(saved["targets"], dim=0)
    outputs_sample = torch.cat(saved["outputs"], dim=0)
    visualize_validation_samples_enhanced(
        inputs_sample, targets_sample, outputs_sample, sample_indices=[0, 1])
    return avg_loss, avg_ssim, avg_psnr


def train_amp(model: nn.Module, train_loader, val_loader, device: torch.device, num_epochs: int, optimizer_input: Optional[optim.Optimizer] = None) -> None:
    model = model.to(device)
    criterion = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    optimizer = optimizer_input or optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    scaler = GradScaler()
    ssim_loss_func = StructuralSimilarityIndexMeasure().to(device)
    best_val_psnr = 0.0
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        checkpoint_path = f"checkpoint_{epoch + 1}.pth"
        model.train()
        running_loss = running_l1 = running_ssim = running_grad = running_mse = 0.0
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, targets in train_loader_tqdm:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            thermal_input = inputs[:, :2, :, :]
            rgb_input = inputs[:, 2:, :, :]
            optimizer.zero_grad()
            with autocast(device_type="cuda" if device.type == "cuda" else None):
                outputs = model(thermal_input, rgb_input)
                outputs = (outputs + 1) / 2
                loss_l1 = criterion(outputs, targets)
                loss_mse = criterion_mse(outputs, targets)
                with torch.no_grad():
                    loss_ssim = 1 - ssim_loss_func(outputs, targets)
                    # simple gradient loss
                    grad_pred_x = outputs[:, :, :, :-1] - outputs[:, :, :, 1:]
                    grad_pred_y = outputs[:, :, :-1, :] - outputs[:, :, 1:, :]
                    grad_tar_x = targets[:, :, :, :-1] - targets[:, :, :, 1:]
                    grad_tar_y = targets[:, :, :-1, :] - targets[:, :, 1:, :]
                    loss_grad = torch.nn.functional.l1_loss(
                        grad_pred_x, grad_tar_x) + torch.nn.functional.l1_loss(grad_pred_y, grad_tar_y)
                loss = loss_l1 + 0.05 * loss_ssim + 0.1 * loss_grad + loss_mse
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            running_l1 += loss_l1.item()
            running_ssim += loss_ssim.item()
            running_grad += loss_grad.item()
            running_mse += loss_mse.item()
            train_loader_tqdm.set_postfix(loss=loss.item(), SSIM=loss_ssim.item(
            ), L1Loss=loss_l1.item(), MSELoss=loss_mse.item())
        epoch_loss = running_loss / len(train_loader)
        epoch_l1_loss = running_l1 / len(train_loader)
        epoch_ssim_loss = running_ssim / len(train_loader)
        epoch_grad_loss = running_grad / len(train_loader)
        epoch_mse_loss = running_mse / len(train_loader)
        print(
            f"Epoch [{epoch + 1}] Average Loss: {epoch_loss:.4f} | Average L1 Loss:{epoch_l1_loss:.4f} | Average SSIM Loss:{epoch_ssim_loss:.4f} | Average Gradient Loss:{epoch_grad_loss:.4f} | Average MSE Loss:{epoch_mse_loss:.4f}"
        )
        val_loss, _, val_psnr = validate(model, val_loader, device)
        scheduler.step()
        if (epoch + 1) % 5 == 0 and val_psnr > best_val_psnr:
            print("Saving new best model checkpoint!")
            best_val_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        torch.cuda.empty_cache()
    print("Training complete.")


def main() -> None:
    cfg = TrainConfig(data_root="/kaggle/input/sih-isro/thermal_degraded1")
    device = torch.device(cfg.device)
    file_list = make_file_list(cfg.data_root)
    train_loader, val_loader, _ = make_dataloaders(
        file_list,
        subset_size=cfg.subset_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    model = CrossModalGDNet(thermal_in_channels=2, optical_in_channels=3,
                            base_channels=cfg.base_channels, scale_factor=cfg.scale_factor).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    train_amp(model, train_loader, val_loader, device, num_epochs=cfg.epochs)


if __name__ == "__main__":
    main()
