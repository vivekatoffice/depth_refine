"""Run depth estimation algorithm on an input RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from transformers import pipeline


def run_depth_estimation(
    image_path: Path,
    output_dir: Path | None = None,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str | None = None,
) -> dict[str, Path]:
    """Run Depth-Anything-V2 inference and save depth visualizations."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if output_dir is None:
        output_dir = image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"Using GPU: {torch.cuda.get_device_name(0)} ({device})")
        else:
            device = "cpu"
            print("Notice: CUDA is not enabled in this Python environment. Running on CPU.")
            print("Tip: Activate the virtual environment to use your GPU (NVIDIA RTX A1000):")
            print(r"     .\.venv\Scripts\python.exe run_depth.py ...")
    else:
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}', but CUDA is not available in this Python environment.\n"
                r"Please run using the virtual environment: .\.venv\Scripts\python.exe run_depth.py"
            )
        print(f"Using device: {device}")

    print(f"Loading model '{model_name}' on device '{device}'...")

    pipe = pipeline("depth-estimation", model=model_name, device=device)

    print(f"Loading input image: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    width, height = raw_image.size
    print(f"Input image resolution: {width}x{height}")

    print("Running depth estimation...")
    result = pipe(raw_image)
    depth_image = result["depth"]  # PIL image
    depth_array = np.array(depth_image, dtype=np.float32)

    # Normalize depth to [0, 1] for visualization
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    norm_depth = (depth_array - depth_min) / (depth_max - depth_min + 1e-8)

    base_stem = image_path.stem

    # 1. Save normalized 8-bit grayscale depth
    depth_gray_path = output_dir / f"{base_stem}_depth_gray.png"
    depth_gray_uint8 = (norm_depth * 255).astype(np.uint8)
    Image.fromarray(depth_gray_uint8).save(depth_gray_path)
    print(f"Saved grayscale depth map: {depth_gray_path}")

    # 2. Save colored depth map (Inferno)
    depth_colored_path = output_dir / f"{base_stem}_depth_colored.png"
    cmap = plt.get_cmap("inferno")
    depth_colored_rgba = (cmap(norm_depth) * 255).astype(np.uint8)
    Image.fromarray(depth_colored_rgba[:, :, :3]).save(depth_colored_path)
    print(f"Saved colored depth map: {depth_colored_path}")

    # 3. Save side-by-side comparison figure
    comparison_path = output_dir / f"{base_stem}_depth_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    axes[0].imshow(raw_image)
    axes[0].set_title("Input RGB Image", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(norm_depth, cmap="inferno")
    axes[1].set_title("Depth Map (Inferno)", fontsize=14, fontweight="bold", pad=10)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Normalized Depth")

    im2 = axes[2].imshow(norm_depth, cmap="plasma")
    axes[2].set_title("Depth Map (Plasma)", fontsize=14, fontweight="bold", pad=10)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Normalized Depth")

    plt.tight_layout()
    fig.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {comparison_path}")

    # 4. Save 16-bit raw depth (standard precision representation)
    depth_16bit_path = output_dir / f"{base_stem}_depth_16bit.png"
    depth_16bit = (norm_depth * 65535.0).astype(np.uint16)
    Image.fromarray(depth_16bit).save(depth_16bit_path)
    print(f"Saved 16-bit raw depth: {depth_16bit_path}")

    return {
        "gray": depth_gray_path,
        "colored": depth_colored_path,
        "comparison": comparison_path,
        "raw_16bit": depth_16bit_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run depth estimation on an image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("images/test.jpg"),
        help="Path to input image (default: images/test.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input image)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face depth estimation model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. cuda, cpu)",
    )
    args = parser.parse_args()

    run_depth_estimation(
        image_path=args.image,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
