"""
Phase 2 — Train a DCGAN for ONE target class
=============================================
Trains a DCGAN on the REAL train-split images of a single pathology class.
One small GAN per class (this is the plan's recommended scope — avoids the
extra complexity of a single conditional GAN).

Only train-split images are used: val/test images are never shown to the
generator, so the classifier is never graded against images the GAN could
have memorized.

Usage:
    python -m src.gan.train_gan --class_name "Pulmonary Hyperinflation" --epochs 300

Outputs:
    checkpoints/gan/<ClassName>/generator.pth     — trained generator
    checkpoints/gan/<ClassName>/discriminator.pth — trained discriminator
    checkpoints/gan/<ClassName>/config.json       — hyperparameters
    runs/gan/<ClassName>/epoch_XXXX.png           — sample grids (watch these
                                                    for mode collapse)

Tips:
    - Watch runs/gan/<ClassName>/ every few hundred iterations.
      If every image in the grid looks identical → mode collapse; restart
      with a lower lr or fewer epochs.
    - 300-500 epochs is a reasonable starting point for 100-300 images.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import get_splits
from src.gan.dcgan import Generator, Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grid_image(tensors, nrow=4, normalize=True, range_=(-1, 1)):
    """Build a numpy grid image from a batch of (1,224,224) tensors."""
    from torchvision.utils import make_grid
    # torchvision >= 0.17 renamed make_grid's ``range`` kwarg to ``value_range``
    # (the old name was removed, crashing every sample-grid save).
    grid = make_grid(tensors, nrow=nrow, normalize=normalize, value_range=range_)
    return grid.detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,C) in [0,1]


def save_grid(generator, z, path, nrow=4):
    """Generate + save a sample grid so the user can spot mode collapse."""
    import cv2
    generator.eval()
    with torch.no_grad():
        fake = generator(z)
    grid_np = make_grid_image(fake, nrow=nrow)
    img = (grid_np * 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    generator.train()


def load_class_images(class_name, max_images=0):
    """
    Load all TRAIN-split images for one class.

    Returns:
        (tensor (N,1,224,224) normalized to [-1,1], count)
    """
    from PIL import Image
    df, label_names, label_vectors, train_mask, val_mask, test_mask = get_splits()

    if class_name not in label_names:
        print(f"ERROR: class '{class_name}' not in label set: {label_names}")
        sys.exit(1)

    class_idx = label_names.index(class_name)
    train_df = df[train_mask].reset_index(drop=True)
    train_vecs = label_vectors[train_mask]
    sel = train_vecs[:, class_idx] == 1.0

    filenames = train_df.loc[sel, "filename"].values
    if max_images and len(filenames) > max_images:
        filenames = filenames[:max_images]
    print(f"  {class_name}: {len(filenames)} real train images for GAN")

    from dataset import IMAGES_DIR
    images = []
    for fn in filenames:
        img = Image.open(IMAGES_DIR / fn).convert("L")
        arr = np.array(img).astype(np.float32)
        # Already 224x224; normalize to [-1, 1] to match generator tanh output
        arr = (arr / 127.5) - 1.0
        images.append(arr)
    data = np.stack(images)[:, None, :, :]  # (N,1,224,224)
    return torch.from_numpy(data).float(), len(filenames)


def train(args):
    print("=" * 60)
    print(f"DCGAN Training — class: {args.class_name}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load real train images for this class only ──
    print("\n--- Loading class images (train split only) ---")
    real_tensor, n_real = load_class_images(args.class_name, args.max_images)
    if n_real < 2:
        print(f"ERROR: only {n_real} image(s) for class '{args.class_name}' — at least 2 "
              f"are required (BatchNorm needs >1 sample per batch).")
        sys.exit(1)
    if n_real < 20:
        print(f"WARNING: only {n_real} images — GAN will likely be unstable.")

    # BatchNorm layers need >1 sample per batch. Drop a trailing 1-image batch,
    # but keep a larger trailing partial batch. Previously drop_last=True was
    # unconditional, so a class with fewer images than batch_size produced ZERO
    # batches per epoch and the epoch-average below crashed with
    # ZeroDivisionError.
    drop_last = (n_real % args.batch_size) == 1 and n_real > args.batch_size
    loader = DataLoader(
        TensorDataset(real_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    # ── 2. Build GAN ──
    generator = Generator(latent_dim=args.latent_dim).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out_dir = Path(args.out_dir) / args.class_name
    runs_dir = Path(args.runs_dir) / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Fixed z for monitoring grids so you can see progress frame-to-frame
    fixed_z = torch.randn(16, args.latent_dim, device=DEVICE)

    print("\n--- Training ---")
    print(f"{'Epoch':>6} | {'D loss':>8} {'G loss':>8} | {'D(real)':>8} {'D(fake)':>8}")
    print("-" * 50)

    total_iters = 0
    for epoch in range(1, args.epochs + 1):
        epoch_d = 0.0
        epoch_g = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        n_batches = 0

        for (real_batch,) in loader:
            batch_size = real_batch.size(0)
            real_batch = real_batch.to(DEVICE)

            # ── Discriminator ──
            opt_d.zero_grad()
            real_labels = torch.full((batch_size, 1), 1.0, device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.0, device=DEVICE)

            out_real = discriminator(real_batch)
            loss_d_real = criterion(out_real, real_labels)

            z = torch.randn(batch_size, args.latent_dim, device=DEVICE)
            fake = generator(z).detach()
            out_fake = discriminator(fake)
            loss_d_fake = criterion(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ── Generator ──
            opt_g.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=DEVICE)
            fake = generator(z)
            out_fake2 = discriminator(fake)
            loss_g = criterion(out_fake2, real_labels)
            loss_g.backward()
            opt_g.step()

            epoch_d += loss_d.item()
            epoch_g += loss_g.item()
            epoch_d_real += out_real.mean().item()
            epoch_d_fake += out_fake.mean().item()
            n_batches += 1
            total_iters += 1

            # Sample grid every sample_every iterations (watch for mode collapse)
            if total_iters % args.sample_every == 0:
                grid_path = runs_dir / f"iter_{total_iters:06d}.png"
                save_grid(generator, fixed_z, grid_path)
                print(f"  [iter {total_iters}] sample grid saved: {grid_path}")

        d_avg = epoch_d / n_batches
        g_avg = epoch_g / n_batches
        dr_avg = epoch_d_real / n_batches
        df_avg = epoch_d_fake / n_batches
        print(f"{epoch:6d} | {d_avg:8.4f} {g_avg:8.4f} | {dr_avg:8.4f} {df_avg:8.4f}")

        # Save periodic checkpoints (every save_every epochs and last epoch)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(generator.state_dict(), out_dir / "generator.pth")
            torch.save(discriminator.state_dict(), out_dir / "discriminator.pth")
            grid_path = runs_dir / f"epoch_{epoch:04d}.png"
            save_grid(generator, fixed_z, grid_path)
            print(f"  >> checkpoint saved: {out_dir / 'generator.pth'}")

    # ── 3. Save final generator + config ──
    torch.save(generator.state_dict(), out_dir / "generator.pth")
    torch.save(discriminator.state_dict(), out_dir / "discriminator.pth")
    config = {
        "class_name": args.class_name,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_real_train_images": n_real,
        "architecture": "dcgan",
        "image_size": 224,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    final_grid = runs_dir / "final.png"
    save_grid(generator, fixed_z, final_grid)
    print("\n" + "=" * 60)
    print(f"DONE. Generator: {out_dir / 'generator.pth'}")
    print(f"      Config:    {out_dir / 'config.json'}")
    print(f"      Final grid: {final_grid}")
    print("Review the sample grids in runs/gan/<class>/ — if all images look")
    print("identical (mode collapse), retrain with a lower lr / fewer epochs.")


def main():
    parser = argparse.ArgumentParser(description="Train a DCGAN for one pathology class")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Pathology class to synthesize (must match label set)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Cap the number of real images used (0 = all)")
    parser.add_argument("--out_dir", type=str, default="checkpoints/gan")
    parser.add_argument("--runs_dir", type=str, default="runs/gan")
    parser.add_argument("--sample_every", type=int, default=200,
                        help="Save a sample grid every N iterations")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save a generator checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()