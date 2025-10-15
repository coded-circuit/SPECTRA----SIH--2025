from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt


def show_images_grid(images_list, titles, figsize=(20, 5), normalize=True, cmap_gray="gray"):
    n = len(images_list)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images_list):
        plt.subplot(1, n, i + 1)
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img.squeeze(0)
                plt.imshow(img, cmap=cmap_gray)
            elif img.shape[0] == 2:
                plt.subplot(1, n + 1, i + 1)
                plt.imshow(img[0], cmap=cmap_gray)
                plt.title(f"{titles[i]} - Channel 1")
                plt.colorbar()
                plt.axis("off")
                plt.subplot(1, n + 1, i + 2)
                plt.imshow(img[1], cmap=cmap_gray)
                plt.title(f"{titles[i]} - Channel 2")
                plt.colorbar()
                plt.axis("off")
                continue
            else:
                img = img.transpose(1, 2, 0)
                plt.imshow(img)
        elif img.ndim == 2:
            plt.imshow(img, cmap=cmap_gray)
        else:
            plt.imshow(img)
        if normalize:
            img = img - np.min(img)
            if np.max(img) > 0:
                img = img / np.max(img)
        plt.axis("off")
        plt.title(titles[i])
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_validation_samples_enhanced(inputs, targets, outputs, sample_indices=None):
    if sample_indices is None:
        sample_indices = [0]
    for idx in sample_indices:
        thermal_in = inputs[idx, :2, :, :]
        rgb_in = inputs[idx, 2:, :, :]
        target_img = targets[idx]
        output_img = outputs[idx]
        output_img = (output_img + 1) / 2
        difference = torch.abs(output_img - target_img)
        images = [thermal_in, rgb_in, target_img, output_img, difference]
        titles = ["Thermal Input", "RGB Input",
                  "Target", "Model Output", "Abs. Difference"]
        show_images_grid(images, titles, figsize=(25, 5))
