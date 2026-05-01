"""
Inference and visual evaluation for the Flow Matching World Model.

Supports multiple ODE solvers:
  - euler: 1st order, fast (10-20 steps sufficient)
  - heun:  2nd order, better quality (10 steps ≈ euler 20 steps)
  - rk4:   4th order, highest quality (5-10 steps)
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.crafter_dataset import CrafterFlowMatchingDataset
from model import FlowMatchingWorldModel

DEVICE = "cuda:0"
CHECKPOINT_PATH = "checkpoints/best.pt"
OUTPUT_DIR = "outputs"
NUM_SAMPLES = 16
ODE_STEPS = 20
SOLVER = "euler"  # "euler", "heun", or "rk4"

# Crafter action names for visualization
ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]


# ---------------------------------------------------------------------------
# ODE Solvers
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(model, cond, num_steps=20):
    """1st order Euler method."""
    B = cond.shape[0]
    x = torch.randn(B, 3, 64, 64, device=cond.device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=cond.device)
        v = model.flow_net(x, t, cond)
        x = x + v * dt
    return x.clamp(0, 1)


@torch.no_grad()
def heun_sample(model, cond, num_steps=20):
    """2nd order Heun's method (improved Euler)."""
    B = cond.shape[0]
    x = torch.randn(B, 3, 64, 64, device=cond.device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_i = torch.full((B,), i * dt, device=cond.device)
        t_next = torch.full((B,), (i + 1) * dt, device=cond.device)

        v1 = model.flow_net(x, t_i, cond)
        x_euler = x + v1 * dt
        v2 = model.flow_net(x_euler, t_next, cond)
        x = x + 0.5 * (v1 + v2) * dt
    return x.clamp(0, 1)


@torch.no_grad()
def rk4_sample(model, cond, num_steps=20):
    """4th order Runge-Kutta method."""
    B = cond.shape[0]
    x = torch.randn(B, 3, 64, 64, device=cond.device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_i = i * dt
        t_mid = (i + 0.5) * dt
        t_next = (i + 1) * dt

        t1 = torch.full((B,), t_i, device=cond.device)
        t2 = torch.full((B,), t_mid, device=cond.device)
        t3 = torch.full((B,), t_next, device=cond.device)

        k1 = model.flow_net(x, t1, cond)
        k2 = model.flow_net(x + 0.5 * dt * k1, t2, cond)
        k3 = model.flow_net(x + 0.5 * dt * k2, t2, cond)
        k4 = model.flow_net(x + dt * k3, t3, cond)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x.clamp(0, 1)


SOLVERS = {"euler": euler_sample, "heun": heun_sample, "rk4": rk4_sample}


def generate(model, s_t, a_t, solver="euler", num_steps=20):
    """Generate s_{t+1} predictions."""
    cond = model.cond_net(s_t, a_t)
    return SOLVERS[solver](model, cond, num_steps)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def to_image(tensor):
    """Convert (C, H, W) tensor [0,1] to PIL Image."""
    arr = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_comparison_grid(s_t_imgs, pred_imgs, gt_imgs, actions, save_path):
    """Create a labeled comparison grid: s_t | predicted s_{t+1} | ground truth s_{t+1}."""
    n = len(s_t_imgs)
    cell_size = 64
    pad = 4
    label_h = 18
    cols = 3
    col_labels = ["s_t (input)", "predicted s_{t+1}", "ground truth"]

    grid_w = cols * (cell_size + pad) + pad
    grid_h = label_h + n * (cell_size + pad + label_h) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Column headers
    for j, label in enumerate(col_labels):
        x = pad + j * (cell_size + pad)
        draw.text((x, 2), label, fill=(200, 200, 200))

    # Rows
    for i in range(n):
        row_y = label_h + i * (cell_size + pad + label_h)
        # Action label
        draw.text((pad, row_y), f"a={actions[i]}", fill=(150, 220, 150))

        img_y = row_y + label_h
        for j, img in enumerate([s_t_imgs[i], pred_imgs[i], gt_imgs[i]]):
            x = pad + j * (cell_size + pad)
            grid.paste(img, (x, img_y))

    grid.save(save_path)
    print(f"Saved comparison grid to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load validation data
    val_ds = CrafterFlowMatchingDataset(
        "/home/chaeun/OTFM-WM/data/crafter_human_dataset/dataset",
        split="val", train_ratio=0.9,
    )

    # Load model (EMA weights)
    model = FlowMatchingWorldModel().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded EMA model from {CHECKPOINT_PATH}")

    # Sample random indices
    rng = np.random.default_rng(42)
    indices = rng.choice(len(val_ds), size=NUM_SAMPLES, replace=False)

    # Collect batch
    s_t_list, a_t_list, gt_list = [], [], []
    for idx in indices:
        s_t, a_t, s_tp1 = val_ds[idx]
        s_t_list.append(s_t)
        a_t_list.append(a_t)
        gt_list.append(s_tp1)

    s_t_batch = torch.stack(s_t_list).to(DEVICE)
    a_t_batch = torch.stack(a_t_list).to(DEVICE)
    gt_batch = torch.stack(gt_list)

    # Generate predictions
    print(f"Generating {NUM_SAMPLES} predictions | solver={SOLVER}, steps={ODE_STEPS}")
    pred_batch = generate(model, s_t_batch, a_t_batch, solver=SOLVER, num_steps=ODE_STEPS)

    # Convert to images
    s_t_imgs = [to_image(s_t_batch[i].cpu()) for i in range(NUM_SAMPLES)]
    pred_imgs = [to_image(pred_batch[i].cpu()) for i in range(NUM_SAMPLES)]
    gt_imgs = [to_image(gt_batch[i]) for i in range(NUM_SAMPLES)]
    actions = [ACTION_NAMES[a_t_list[i].item()] for i in range(NUM_SAMPLES)]

    # Save comparison grid
    make_comparison_grid(s_t_imgs, pred_imgs, gt_imgs, actions,
                         f"{OUTPUT_DIR}/comparison_{SOLVER}_{ODE_STEPS}steps.png")

    # Save individual samples
    for i in range(min(4, NUM_SAMPLES)):
        s_t_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_st.png")
        pred_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_pred.png")
        gt_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_gt.png")
        print(f"  Sample {i}: action={actions[i]}")

    # Quantitative metrics
    pred_cpu = pred_batch.cpu()
    mse = ((pred_cpu - gt_batch) ** 2).mean().item()
    psnr = -10 * np.log10(mse + 1e-8)
    print(f"\nPixel MSE: {mse:.5f}")
    print(f"PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
