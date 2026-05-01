"""
V2: Inference for Conditional OT Flow Matching World Model.

Key change: ODE starts from s_t (not noise) and flows to predicted s_{t+1}.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw

from data.crafter_dataset import CrafterFlowMatchingDataset
from model_v2 import FlowMatchingWorldModelV2

DEVICE = "cuda:0"
CHECKPOINT_PATH = "checkpoints_v2/best.pt"
OUTPUT_DIR = "outputs_v2"
NUM_SAMPLES = 16
ODE_STEPS = 20
SOLVER = "euler"

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]


# ---------------------------------------------------------------------------
# ODE Solvers (starting from s_t, not noise)
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(model, s_t, a_t, num_steps=20):
    """Euler ODE: s_t -> predicted s_{t+1}."""
    B = s_t.shape[0]
    x = s_t.clone()  # start from s_t, NOT noise
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=s_t.device)
        v = model(x, t, s_t, a_t)
        x = x + v * dt
    return x.clamp(0, 1)


@torch.no_grad()
def heun_sample(model, s_t, a_t, num_steps=20):
    """Heun's method (2nd order): s_t -> predicted s_{t+1}."""
    B = s_t.shape[0]
    x = s_t.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_i = torch.full((B,), i * dt, device=s_t.device)
        t_next = torch.full((B,), (i + 1) * dt, device=s_t.device)
        v1 = model(x, t_i, s_t, a_t)
        x_euler = x + v1 * dt
        v2 = model(x_euler, t_next, s_t, a_t)
        x = x + 0.5 * (v1 + v2) * dt
    return x.clamp(0, 1)


@torch.no_grad()
def rk4_sample(model, s_t, a_t, num_steps=20):
    """RK4 (4th order): s_t -> predicted s_{t+1}."""
    B = s_t.shape[0]
    x = s_t.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t1 = torch.full((B,), i * dt, device=s_t.device)
        t2 = torch.full((B,), (i + 0.5) * dt, device=s_t.device)
        t3 = torch.full((B,), (i + 1) * dt, device=s_t.device)
        k1 = model(x, t1, s_t, a_t)
        k2 = model(x + 0.5 * dt * k1, t2, s_t, a_t)
        k3 = model(x + 0.5 * dt * k2, t2, s_t, a_t)
        k4 = model(x + dt * k3, t3, s_t, a_t)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x.clamp(0, 1)


SOLVERS = {"euler": euler_sample, "heun": heun_sample, "rk4": rk4_sample}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def to_image(tensor):
    arr = (tensor.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_comparison_grid(s_t_imgs, pred_imgs, gt_imgs, actions, save_path):
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

    for j, label in enumerate(col_labels):
        x = pad + j * (cell_size + pad)
        draw.text((x, 2), label, fill=(200, 200, 200))

    for i in range(n):
        row_y = label_h + i * (cell_size + pad + label_h)
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

    val_ds = CrafterFlowMatchingDataset(
        "/home/chaeun/OTFM-WM/data/crafter_human_dataset/dataset",
        split="val", train_ratio=0.9,
    )

    model = FlowMatchingWorldModelV2().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded EMA model from {CHECKPOINT_PATH}")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(val_ds), size=NUM_SAMPLES, replace=False)

    s_t_list, a_t_list, gt_list = [], [], []
    for idx in indices:
        s_t, a_t, s_tp1 = val_ds[idx]
        s_t_list.append(s_t)
        a_t_list.append(a_t)
        gt_list.append(s_tp1)

    s_t_batch = torch.stack(s_t_list).to(DEVICE)
    a_t_batch = torch.stack(a_t_list).to(DEVICE)
    gt_batch = torch.stack(gt_list)

    print(f"Generating {NUM_SAMPLES} predictions | solver={SOLVER}, steps={ODE_STEPS}")
    solve_fn = SOLVERS[SOLVER]
    pred_batch = solve_fn(model, s_t_batch, a_t_batch, num_steps=ODE_STEPS)

    s_t_imgs = [to_image(s_t_batch[i].cpu()) for i in range(NUM_SAMPLES)]
    pred_imgs = [to_image(pred_batch[i].cpu()) for i in range(NUM_SAMPLES)]
    gt_imgs = [to_image(gt_batch[i]) for i in range(NUM_SAMPLES)]
    actions = [ACTION_NAMES[a_t_list[i].item()] for i in range(NUM_SAMPLES)]

    make_comparison_grid(s_t_imgs, pred_imgs, gt_imgs, actions,
                         f"{OUTPUT_DIR}/comparison_{SOLVER}_{ODE_STEPS}steps.png")

    for i in range(min(4, NUM_SAMPLES)):
        s_t_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_st.png")
        pred_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_pred.png")
        gt_imgs[i].save(f"{OUTPUT_DIR}/sample_{i}_gt.png")
        print(f"  Sample {i}: action={actions[i]}")

    pred_cpu = pred_batch.cpu()
    mse = ((pred_cpu - gt_batch) ** 2).mean().item()
    psnr = -10 * np.log10(mse + 1e-8)
    print(f"\nPixel MSE: {mse:.5f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"\n(V1 baseline: MSE=0.02739, PSNR=15.62 dB)")


if __name__ == "__main__":
    main()
