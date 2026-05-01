"""
OT Flow Matching training script for Crafter World Model.

Training procedure:
  1. Sample (s_t, a_t, s_{t+1}) from dataset
  2. x_1 = s_{t+1}, x_0 ~ N(0, I)
  3. t ~ U(0, 1)
  4. x_t = (1 - t) * x_0 + t * x_1  (OT interpolation)
  5. Target vector field: u_t = x_1 - x_0
  6. Loss = MSE(v_theta(x_t, t, c), u_t)

Features:
  - Linear warmup + cosine annealing scheduler
  - EMA (Exponential Moving Average) of model weights
  - Batch size 256 optimized for RTX 3090
"""

import copy
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data.crafter_dataset import build_dataloaders
from model import FlowMatchingWorldModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda:0"
BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 200
WARMUP_STEPS = 1000
EMA_DECAY = 0.999
SAVE_EVERY = 10        # Save checkpoint every N epochs
LOG_EVERY = 50         # Print loss every N steps
CHECKPOINT_DIR = "checkpoints"
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.lerp_(m_param, 1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Warmup + Cosine Scheduler
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, ema, epoch, global_step, total_steps):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, (s_t, a_t, s_tp1) in enumerate(loader):
        s_t = s_t.to(DEVICE)
        a_t = a_t.to(DEVICE)
        s_tp1 = s_tp1.to(DEVICE)

        B = s_t.shape[0]

        # OT Flow Matching setup
        x_1 = s_tp1                                         # target: next state
        x_0 = torch.randn_like(x_1)                         # prior: Gaussian noise
        t = torch.rand(B, device=DEVICE)                     # time ~ U(0, 1)

        # OT interpolation
        t_expand = t[:, None, None, None]                    # (B, 1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1         # noisy interpolation

        # Target vector field
        u_t = x_1 - x_0

        # Model prediction
        v_pred = model(x_t, t, s_t, a_t)

        # MSE loss
        loss = F.mse_loss(v_pred, u_t)

        # LR scheduling (step-level warmup + cosine)
        lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update
        ema.update(model)

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if step % LOG_EVERY == 0:
            print(f"  [Epoch {epoch}] step {step}/{len(loader)}, "
                  f"loss={loss.item():.5f}, lr={lr:.2e}")

    return total_loss / n_batches, global_step


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for s_t, a_t, s_tp1 in loader:
        s_t = s_t.to(DEVICE)
        a_t = a_t.to(DEVICE)
        s_tp1 = s_tp1.to(DEVICE)

        B = s_t.shape[0]
        x_1 = s_tp1
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=DEVICE)

        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0

        v_pred = model(x_t, t, s_t, a_t)
        loss = F.mse_loss(v_pred, u_t)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    print("Building model...")
    model = FlowMatchingWorldModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = EMA(model, decay=EMA_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, "
          f"Warmup: {WARMUP_STEPS} steps")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, ema, epoch, global_step, total_steps
        )

        # Validate with EMA model
        val_loss = validate(ema.shadow, val_loader)
        val_loss_raw = validate(model, val_loader)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS} | train={train_loss:.5f} | "
              f"val_ema={val_loss:.5f} | val_raw={val_loss_raw:.5f} | "
              f"lr={lr:.2e} | time={elapsed:.1f}s")

        # Save best EMA model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema.state_dict(), f"{CHECKPOINT_DIR}/best.pt")
            print(f"  -> New best val_loss(EMA)={val_loss:.5f}, saved.")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, f"{CHECKPOINT_DIR}/epoch_{epoch:04d}.pt")

    print(f"\nTraining done. Best val_loss(EMA)={best_val_loss:.5f}")


if __name__ == "__main__":
    main()
