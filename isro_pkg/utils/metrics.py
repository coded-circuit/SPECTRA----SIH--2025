from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def safe_psnr(true_img: np.ndarray, pred_img: np.ndarray, data_range: float) -> float:
    mse = np.mean((true_img - pred_img) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((data_range**2) / mse)


@torch.no_grad()
def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    ssim_total = 0.0
    psnr_total = 0.0
    count = 0

    # scale from [-1,1] to [0,1]
    output = (output + 1) / 2
    output = torch.clamp(output, 0, 1)

    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    for i in range(output_np.shape[0]):
        out_img = np.transpose(output_np[i], (1, 2, 0))
        tgt_img = np.transpose(target_np[i], (1, 2, 0))

        height, width = out_img.shape[:2]
        min_dim = min(height, width)
        win_size = 7 if min_dim >= 7 else (
            min_dim if min_dim % 2 == 1 else max(min_dim - 1, 3))
        if min_dim < win_size or win_size < 3:
            continue

        data_range = out_img.max() - out_img.min()
        if data_range == 0:
            data_range = 1e-6

        ssim_val = ssim(tgt_img, out_img, channel_axis=-1,
                        data_range=data_range, win_size=win_size)
        psnr_val = psnr(tgt_img, out_img, data_range=data_range)
        ssim_total += ssim_val
        psnr_total += psnr_val
        count += 1

    if count == 0:
        return 0.0, 0.0
    return ssim_total / count, psnr_total / count
