"""
Phase 3 — Generate + quality-check synthetic images
====================================================
Loads a trained DCGAN generator for one class and writes synthetic chest
X-ray PNGs into data/synthetic/<ClassName>/, plus:

    data/synthetic/_review/<ClassName>.png   — contact sheet of ALL generated
                                               images for visual review
    data/synthetic/<ClassName>/manifest.csv  — list of generated files

The review contact sheet is stored OUTSIDE the class folder so it can never
be mistaken for a training image (the dataset loader scans only
data/synthetic/<ClassName>/*.png).

QUALITY GATE (non-negotiable): visually inspect the contact sheet BEFORE
training with these images. Look for:
    - obvious artifacts / unrealistic anatomy
    - images that look clean but encode statistical shortcuts
Delete bad images (or the whole batch and regenerate) before Phase 4.

Usage:
    python -m src.gan.generate --class_name "Pulmonary Hyperinflation" \
        --num_images 300

    # Or scale relative to the real count (2-3x recommended, not 10x):
    python -m src.gan.generate --class_name "Pulmonary Hyperinflation" \
        --multiplier 2.5

    # Rebuild manifest + contact sheet only (e.g. after deleting bad images):
    python -m src.gan.generate --class_name "Pulmonary Hyperinflation" \
        --rebuild-only
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gan.dcgan import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYNTH_ROOT = Path("data/synthetic")


def load_generator(class_name, generator_path, out_dir):
    """Load generator weights + config; returns (generator, latent_dim)."""
    gen_path = Path(generator_path)
    if not gen_path.exists():
        gen_path = Path(out_dir) / class_name / "generator.pth"
    if not gen_path.exists():
        print(f"ERROR: generator not found: {gen_path}")
        sys.exit(1)

    config_path = Path(out_dir) / class_name / "config.json"
    latent_dim = 100
    if config_path.exists():
        with open(config_path) as f:
            latent_dim = json.load(f).get("latent_dim", 100)

    generator = Generator(latent_dim=latent_dim).to(DEVICE)
    generator.load_state_dict(torch.load(gen_path, map_location=DEVICE, weights_only=True))
    generator.eval()
    print(f"Loaded generator: {gen_path} (latent_dim={latent_dim})")
    return generator, latent_dim


def save_contact_sheet(png_paths, out_png, cols=6, thumb=224):
    """Tile all synthetic images into one contact sheet for visual review."""
    thumbs = []
    for p in png_paths:
        img = Image.open(p).convert("L")
        thumbs.append(img)
    if not thumbs:
        print("  (no images to tile)")
        return

    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("L", (cols * thumb, rows * thumb), 0)
    for i, img in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(img, (c * thumb, r * thumb))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"  Contact sheet: {out_png} ({len(thumbs)} images)")


def write_manifest(class_dir, class_name):
    """Write manifest.csv listing the current synthetic PNGs for a class."""
    pngs = sorted(class_dir.glob("*.png"))
    with open(class_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_name"])
        for p in pngs:
            writer.writerow([p.name, class_name])
    print(f"  Manifest: {class_dir / 'manifest.csv'} ({len(pngs)} images)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic chest X-rays from a trained GAN")
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=0,
                        help="Exact number of images to generate (0 = use --multiplier)")
    parser.add_argument("--multiplier", type=float, default=2.0,
                        help="Generate multiplier x the real train count (default 2.0)")
    parser.add_argument("--generator", type=str, default="",
                        help="Path to generator.pth (default: checkpoints/gan/<class>/generator.pth)")
    parser.add_argument("--out_dir", type=str, default="checkpoints/gan",
                        help="Where the GAN checkpoints live")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Only rebuild manifest + contact sheet from existing PNGs")
    args = parser.parse_args()

    class_dir = SYNTH_ROOT / args.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_only:
        pngs = sorted(class_dir.glob("*.png"))
        write_manifest(class_dir, args.class_name)
        save_contact_sheet(pngs, SYNTH_ROOT / "_review" / f"{args.class_name}.png")
        print("Rebuild done. Review the contact sheet before training.")
        return

    # How many images to generate?
    if args.num_images > 0:
        n = args.num_images
        print(f"Generating {n} images (explicit --num_images)")
    else:
        # Count real train images for this class
        from src.gan.train_gan import load_class_images
        _, n_real = load_class_images(args.class_name)
        n = max(1, int(round(n_real * args.multiplier)))
        print(f"Generating {n} images = {args.multiplier}x real count ({n_real})")

    generator, latent_dim = load_generator(args.class_name, args.generator, args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    png_paths = []
    batch = 64
    generated = 0
    with torch.no_grad():
        while generated < n:
            z = torch.randn(min(batch, n - generated), latent_dim, device=DEVICE)
            fake = generator(z)  # (B,1,224,224) in [-1,1]
            imgs = (fake.cpu().numpy().squeeze(1) + 1.0) * 127.5
            imgs = np.clip(imgs, 0, 255).astype(np.uint8)
            for i in range(imgs.shape[0]):
                p = class_dir / f"syn_{generated + i + 1:05d}.png"
                Image.fromarray(imgs[i], mode="L").save(p)
                png_paths.append(p)
            generated += imgs.shape[0]
            print(f"  generated {generated}/{n}")

    write_manifest(class_dir, args.class_name)
    save_contact_sheet(png_paths, SYNTH_ROOT / "_review" / f"{args.class_name}.png")

    print("\n" + "=" * 60)
    print(f"Generated {generated} synthetic images in {class_dir}")
    print(f"CONTACT SHEET for review: {SYNTH_ROOT / '_review' / f'{args.class_name}.png'}")
    print("VISUALLY REVIEW THE CONTACT SHEET before training with these images!")
    print("Delete bad PNGs from the class folder, then re-run with --rebuild-only")
    print("to refresh the manifest + contact sheet.")


if __name__ == "__main__":
    main()