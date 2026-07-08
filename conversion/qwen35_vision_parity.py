"""Compare fixed-grid Qwen3.5 PyTorch vision features with Core ML."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from build_qwen35_vision import FixedGridQwen35Vision, load_vision


def patchify(image_path: Path, image_height: int,
             image_width: int | None = None) -> np.ndarray:
    image_width = image_width or image_height
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image = image.resize((image_width, image_height), Image.Resampling.BICUBIC)
    chw = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    chw = (chw - 0.5) / 0.5
    frames = np.stack((chw, chw), axis=0)  # temporal patch size = 2
    patch = 16
    merge = 2
    grouped = frames.reshape(
        2, 3,
        image_height // (patch * merge), merge, patch,
        image_width // (patch * merge), merge, patch,
    )
    rows = grouped.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return rows.reshape(-1, 3 * 2 * patch * patch).astype(np.float16)


def cosine_rows(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)
    numerator = (reference * candidate).sum(axis=-1)
    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(candidate, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--coreml", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    args = parser.parse_args()

    image_height = args.image_height or args.image_size
    image_width = args.image_width or args.image_size
    pixels = patchify(Path(args.image), image_height, image_width)
    vision = load_vision(Path(args.model_dir))
    reference_model = FixedGridQwen35Vision(
        vision, image_height, image_width).eval().float()
    with torch.no_grad():
        reference = reference_model(torch.from_numpy(pixels).float()).cpu().numpy()
    print(f"reference shape={reference.shape} norm={np.linalg.norm(reference):.4f}")

    for name, units in (
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
    ):
        model = ct.models.MLModel(args.coreml, compute_units=units)
        prediction = model.predict({"pixel_values": pixels})["image_features"]
        prediction = np.asarray(prediction).reshape(reference.shape)
        row_cos = cosine_rows(reference, prediction)
        overall = cosine_rows(reference.reshape(1, -1), prediction.reshape(1, -1))[0]
        print(
            f"{name} overall_cos={overall:.6f} "
            f"row_cos_min={row_cos.min():.6f} "
            f"row_cos_median={np.median(row_cos):.6f} "
            f"max_abs={np.max(np.abs(reference - prediction)):.6f}")


if __name__ == "__main__":
    main()
