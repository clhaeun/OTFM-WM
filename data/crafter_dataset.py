"""
PyTorch Dataset for Crafter Human Dataset.
Provides (s_t, a_t, s_{t+1}) tuples for world model training.
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CrafterFlowMatchingDataset(Dataset):
    """
    Loads all episodes from the crafter_human_dataset and
    returns (s_t, a_t, s_{t+1}) transitions for flow matching world model training.

    Images are normalized to [0, 1] float32 and channel-first (C, H, W).
    Actions are returned as integer scalars for nn.Embedding.
    """

    NUM_ACTIONS = 17  # Crafter has 17 discrete actions (0-16)

    def __init__(self, data_dir: str, split: str = "train", train_ratio: float = 0.9):
        super().__init__()
        npz_files = sorted(glob.glob(f"{data_dir}/*.npz"))
        assert len(npz_files) > 0, f"No .npz files found in {data_dir}"

        # Split episodes into train/val
        n_train = int(len(npz_files) * train_ratio)
        if split == "train":
            npz_files = npz_files[:n_train]
        else:
            npz_files = npz_files[n_train:]

        # Load all transitions into memory (~62k samples, ~1.5GB, fits in RAM)
        images_list = []
        actions_list = []

        for f in npz_files:
            data = np.load(f)
            imgs = data["image"]       # (T, 64, 64, 3) uint8
            acts = data["action"]      # (T,) int64

            # Create consecutive pairs: (s_t, a_t) -> s_{t+1}
            # Skip transitions across episode boundaries (last step has no next)
            images_list.append(imgs)    # will index [:-1] and [1:]
            actions_list.append(acts)
            data.close()

        # Concatenate and create transition pairs
        self.s_t = []
        self.a_t = []
        self.s_tp1 = []

        for imgs, acts in zip(images_list, actions_list):
            self.s_t.append(imgs[:-1])
            self.a_t.append(acts[:-1])
            self.s_tp1.append(imgs[1:])

        self.s_t = np.concatenate(self.s_t, axis=0)      # (N, 64, 64, 3)
        self.a_t = np.concatenate(self.a_t, axis=0)       # (N,)
        self.s_tp1 = np.concatenate(self.s_tp1, axis=0)   # (N, 64, 64, 3)

        # Detect actual number of actions from data
        actual_max = int(self.a_t.max())
        if actual_max + 1 > self.NUM_ACTIONS:
            self.NUM_ACTIONS = actual_max + 1

        print(f"[CrafterDataset] split={split}, transitions={len(self.s_t)}, "
              f"num_actions={self.NUM_ACTIONS}")

    def __len__(self):
        return len(self.s_t)

    def __getitem__(self, idx):
        # Image: uint8 [0,255] -> float32 [0,1], HWC -> CHW
        s_t = torch.from_numpy(self.s_t[idx]).permute(2, 0, 1).float() / 255.0
        s_tp1 = torch.from_numpy(self.s_tp1[idx]).permute(2, 0, 1).float() / 255.0

        # Action: integer scalar for nn.Embedding
        a_t = torch.tensor(int(self.a_t[idx]), dtype=torch.long)

        return s_t, a_t, s_tp1


def build_dataloaders(
    data_dir: str = "/home/chaeun/OTFM-WM/data/crafter_human_dataset/dataset",
    batch_size: int = 64,
    train_ratio: float = 0.9,
    num_workers: int = 4,
):
    train_ds = CrafterFlowMatchingDataset(data_dir, split="train", train_ratio=train_ratio)
    val_ds = CrafterFlowMatchingDataset(data_dir, split="val", train_ratio=train_ratio)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(batch_size=64)
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    s_t, a_t, s_tp1 = next(iter(train_loader))
    print(f"s_t:   {s_t.shape}, dtype={s_t.dtype}, range=[{s_t.min():.2f}, {s_t.max():.2f}]")
    print(f"a_t:   {a_t.shape}, dtype={a_t.dtype}, range=[{a_t.min()}, {a_t.max()}]")
    print(f"s_tp1: {s_tp1.shape}, dtype={s_tp1.dtype}, range=[{s_tp1.min():.2f}, {s_tp1.max():.2f}]")
