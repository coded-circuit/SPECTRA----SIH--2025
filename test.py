from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from isro_pkg.data.datasets import make_dataloaders, make_file_list
from isro_pkg.models.gdnet import CrossModalGDNet
from isro_pkg.utils.metrics import compute_metrics
from isro_pkg.utils.viz import visualize_validation_samples_enhanced


@dataclass
class TestConfig:
    data_root: str
    subset_size: Optional[int] = None  # evaluate full test split by default
    batch_size: int = 32
    num_workers: int = 4
    base_channels: int = 128
    scale_factor: int = 2
    checkpoint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.L1Loss()
    total_samples = 0
    loss_accum = 0.0
    ssim_accum = 0.0
    psnr_accum = 0.0

    test_tqdm = tqdm(test_loader, desc="Test", leave=False)
    saved = {"inputs": [], "targets": [], "outputs": []}

    for batch_idx, (inputs, targets) in enumerate(test_tqdm):
        inputs = inputs.to(device)
        targets = targets.to(device)
        thermal_in = inputs[:, :2, :, :]
        rgb_in = inputs[:, 2:, :, :]
        outputs = model(thermal_in, rgb_in)
        loss = criterion(outputs, targets)

        bsz = inputs.size(0)
        total_samples += bsz
        loss_accum += loss.item() * bsz

        batch_ssim, batch_psnr = compute_metrics(outputs, targets)
        ssim_accum += batch_ssim * bsz
        psnr_accum += batch_psnr * bsz

        if batch_idx == 0:
            saved["inputs"].append(inputs.cpu())
            saved["targets"].append(targets.cpu())
            saved["outputs"].append(outputs.cpu())

        test_tqdm.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "SSIM": f"{batch_ssim:.4f}",
            "PSNR": f"{batch_psnr:.2f}",
        })

    avg_loss = loss_accum / max(total_samples, 1)
    avg_ssim = ssim_accum / max(total_samples, 1)
    avg_psnr = psnr_accum / max(total_samples, 1)

    print(
        f"Test - Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

    if saved["inputs"]:
        inputs_sample = torch.cat(saved["inputs"], dim=0)
        targets_sample = torch.cat(saved["targets"], dim=0)
        outputs_sample = torch.cat(saved["outputs"], dim=0)
        visualize_validation_samples_enhanced(
            inputs_sample, targets_sample, outputs_sample, sample_indices=[0, 1])

    return avg_loss, avg_ssim, avg_psnr


def build_model(cfg: TestConfig, device: torch.device) -> nn.Module:
    model = CrossModalGDNet(
        thermal_in_channels=2,
        optical_in_channels=3,
        base_channels=cfg.base_channels,
        scale_factor=cfg.scale_factor,
    ).to(device)
    if cfg.checkpoint:
        print(f"Loading checkpoint from: {cfg.checkpoint}")
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(
            state["model_state_dict"] if "model_state_dict" in state else state)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate CrossModalGDNet on test set")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing the .tif files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to load")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Optional subset size for faster testing")
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--scale_factor", type=int, default=2)
    args = parser.parse_args()
    return TestConfig(
        data_root=args.data_root,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        scale_factor=args.scale_factor,
        checkpoint=args.checkpoint,
    )


def main() -> None:
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = make_file_list(cfg.data_root)
    _, _, test_loader = make_dataloaders(
        files,
        subset_size=cfg.subset_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg, device)
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
