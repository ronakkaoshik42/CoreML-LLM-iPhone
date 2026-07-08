"""Locate Qwen3.5 vision ANE drift with a short prefix of vision layers."""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from build_qwen35_vision import load_vision
from build_qwen3_vl_2b_vision import _full_vision_attn_forward
from qwen35_vision_parity import cosine_rows, patchify


class VisionPrefix(nn.Module):
    def __init__(self, vision, image_size: int, layers: int):
        super().__init__()
        self.patch_embed = vision.patch_embed
        self.blocks = nn.ModuleList(vision.blocks[:layers])
        for block in self.blocks:
            block.attn.forward = types.MethodType(_full_vision_attn_forward, block.attn)
        grid_side = image_size // vision.config.patch_size
        self.seq_len = grid_side * grid_side
        grid = torch.tensor([[1, grid_side, grid_side]], dtype=torch.long)
        with torch.no_grad():
            pos = vision.fast_pos_embed_interpolate(grid)
            rotary = vision.rot_pos_emb(grid).reshape(self.seq_len, -1)
            rotary = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer("pos", pos, persistent=False)
        self.register_buffer("cos", rotary.cos(), persistent=False)
        self.register_buffer("sin", rotary.sin(), persistent=False)

    def forward(self, pixels):
        hidden = self.patch_embed(pixels) + self.pos
        cu = torch.tensor([0, self.seq_len], dtype=torch.int32, device=hidden.device)
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=(self.cos, self.sin))
        return hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()

    report = {"stage": "start", "layers": args.layers, "results": {}}

    def save_report() -> None:
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    save_report()

    pixels = patchify(Path(args.image), args.image_size)
    report["stage"] = "loading_reference"
    save_report()
    model = VisionPrefix(load_vision(Path(args.model_dir)), args.image_size, args.layers).eval().float()
    with torch.no_grad():
        reference = model(torch.from_numpy(pixels).float()).numpy()
    report["stage"] = "reference_ready"
    save_report()
    if not args.reuse_existing:
        traced = torch.jit.trace(
            model, torch.zeros_like(torch.from_numpy(pixels).float()), strict=False)
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="pixel_values", shape=pixels.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="hidden", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        converted.save(args.out)
        report["stage"] = "converted"
        save_report()
    for name, units in (("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)):
        report["stage"] = f"running_{name.lower()}"
        save_report()
        coreml = ct.models.MLModel(args.out, compute_units=units)
        prediction = np.asarray(coreml.predict({"pixel_values": pixels})["hidden"]).reshape(reference.shape)
        rows = cosine_rows(reference, prediction)
        overall = cosine_rows(reference.reshape(1, -1), prediction.reshape(1, -1))[0]
        report["results"][name] = {
            "overall": float(overall),
            "row_min": float(rows.min()),
            "row_median": float(np.median(rows)),
        }
        save_report()
        print(f"{name} layers={args.layers} overall={overall:.6f} min={rows.min():.6f} median={np.median(rows):.6f}")
    report["stage"] = "complete"
    save_report()


if __name__ == "__main__":
    main()
