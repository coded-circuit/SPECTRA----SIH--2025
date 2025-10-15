import glob
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class LandsatPatchDataset(Dataset):
    """Dataset that extracts fixed-size patches from Landsat GeoTIFFs.

    The dataset creates an index map of (file_index, top, left) for all
    possible windows given the desired patch size and stride.
    """

    def __init__(
        self,
        tif_paths: Sequence[str],
        patch_size: int = 128,
        stride: int = 128,
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                                      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        self.tif_paths = list(tif_paths)
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms

        self.index_map: List[Tuple[int, int, int]] = []
        for i, path in enumerate(self.tif_paths):
            with rasterio.open(path) as src:
                height, width = src.height, src.width
            for top in range(0, height - patch_size + 1, stride):
                for left in range(0, width - patch_size + 1, stride):
                    self.index_map.append((i, top, left))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, top, left = self.index_map[idx]
        path = self.tif_paths[file_idx]

        # Bands to read: [2,3,4,10,11,12,13]
        bands_to_read = [2, 3, 4, 10, 11, 12, 13]

        with rasterio.open(path) as src:
            window = rasterio.windows.Window(
                left, top, self.patch_size, self.patch_size)
            patch_stack = np.stack([src.read(band, window=window)
                                   for band in bands_to_read])

        # RGB to tensor, reorder from BGR->RGB
        rgb = patch_stack[0:3][[2, 1, 0], :, :]
        rgb = rgb.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).float().unsqueeze(0)
        rgb_resized = F.interpolate(
            rgb_tensor,
            size=(self.patch_size // 2, self.patch_size // 2),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Thermal LR (2 channels) to tensor
        thermal_lr = patch_stack[5:7]
        thermal_lr = thermal_lr.astype(np.float32) / 255.0
        thermal_lr_tensor = torch.from_numpy(thermal_lr).float().unsqueeze(0)
        thermal_lr_resized = F.interpolate(
            thermal_lr_tensor,
            size=(self.patch_size // 2, self.patch_size // 2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Thermal HR (target, 2 channels)
        thermal_hr = patch_stack[3:5]
        thermal_hr = thermal_hr.astype(np.float32) / 255.0
        thermal_hr_tensor = torch.from_numpy(thermal_hr).float()

        # Optional augmentation
        if self.transforms:
            rgb_resized, thermal_lr_resized, thermal_hr_tensor = self.transforms(
                rgb_resized, thermal_lr_resized, thermal_hr_tensor
            )

        input_tensor = torch.cat([thermal_lr_resized, rgb_resized], dim=0)
        return input_tensor, thermal_hr_tensor


def make_file_list(input_root: str) -> List[str]:
    """Recursively collect all .tif files under the given root."""
    return glob.glob(f"{input_root}/**/*.tif", recursive=True)


def make_dataloaders(
    tif_paths: Sequence[str],
    subset_size: Optional[int] = 30000,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    patch_size: int = 128,
    stride: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders mirroring the notebook setup."""
    dataset = LandsatPatchDataset(
        tif_paths, patch_size=patch_size, stride=stride)

    if subset_size is not None:
        dataset_size = len(dataset)
        indices = np.random.permutation(dataset_size)[: subset_size]
        dataset = Subset(dataset, indices)

    total = len(dataset)
    train_size = int(train_frac * total)
    val_size = int(val_frac * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
