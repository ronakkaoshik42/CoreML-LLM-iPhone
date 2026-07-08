"""Convert the Qwen3.5 vision tower + merger to a fixed-grid Core ML model.

The language model remains in llama.cpp. This artifact accepts the same
pre-patchified fp16 pixels as Qwen3.5's Hugging Face processor and emits the
merged 2560-wide visual embeddings consumed by the text model.

Only the official shard containing ``model.visual.*`` is required; the 4B
text weights are intentionally not loaded.
"""
from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from build_qwen3_vl_2b_vision import _audit_ane, _full_vision_attn_forward


class FixedGridQwen35Vision(nn.Module):
    def __init__(self, vision: Qwen3_5VisionModel, image_height: int,
                 image_width: int | None = None):
        super().__init__()
        cfg = vision.config
        image_width = image_width or image_height
        if image_height % cfg.patch_size or image_width % cfg.patch_size:
            raise ValueError("image dimensions must be divisible by patch size")
        self.patch_size = cfg.patch_size
        self.spatial_merge_size = cfg.spatial_merge_size
        self.grid_h = image_height // cfg.patch_size
        self.grid_w = image_width // cfg.patch_size
        self.seq_len = self.grid_h * self.grid_w
        self.patch_embed = vision.patch_embed
        self.blocks = vision.blocks
        self.merger = vision.merger

        for block in self.blocks:
            block.attn.forward = types.MethodType(
                _full_vision_attn_forward, block.attn)

        grid = torch.tensor([[1, self.grid_h, self.grid_w]], dtype=torch.long)
        with torch.no_grad():
            pos = vision.fast_pos_embed_interpolate(grid)
            rotary = vision.rot_pos_emb(grid).reshape(self.seq_len, -1)
            rotary = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer("pos_embed_fixed", pos, persistent=False)
        self.register_buffer("rot_cos", rotary.cos(), persistent=False)
        self.register_buffer("rot_sin", rotary.sin(), persistent=False)

    def forward(self, pixel_values):
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self.pos_embed_fixed
        cu_seqlens = torch.tensor(
            [0, self.seq_len], dtype=torch.int32, device=hidden.device)
        positions = (self.rot_cos, self.rot_sin)
        for block in self.blocks:
            hidden = block(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=positions,
            )
        return self.merger(hidden)


def load_vision(model_dir: Path) -> Qwen3_5VisionModel:
    cfg = Qwen3_5VisionConfig.from_pretrained(model_dir)
    vision = Qwen3_5VisionModel(cfg)
    shard = model_dir / "model.safetensors-00002-of-00002.safetensors"
    state = load_file(str(shard), device="cpu")
    prefix = "model.visual."
    visual_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    missing, unexpected = vision.load_state_dict(visual_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"vision state mismatch: missing={missing} unexpected={unexpected}")
    return vision.eval().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_height = args.image_height or args.image_size
    image_width = args.image_width or args.image_size
    output_name = (
        f"qwen35_vision_{image_height}.mlpackage"
        if image_width == image_height
        else f"qwen35_vision_{image_width}x{image_height}.mlpackage"
    )
    output = out_dir / output_name
    if output.exists():
        raise FileExistsError(f"refusing to replace existing model: {output}")

    started = time.time()
    vision = load_vision(model_dir)
    model = FixedGridQwen35Vision(
        vision, image_height, image_width).eval().float()
    del vision
    patch_flat = 3 * 2 * model.patch_size * model.patch_size
    example = torch.zeros(model.seq_len, patch_flat, dtype=torch.float32)
    print(
        f"grid={model.grid_h}x{model.grid_w} patches={model.seq_len} "
        f"tokens={model.seq_len // model.spatial_merge_size**2}")

    traced = torch.jit.trace(model, example, strict=False)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(
            name="pixel_values",
            shape=example.shape,
            dtype=np.float16,
        )],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    converted.save(str(output))
    size_mb = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    ) / 1e6
    print(f"saved={output} size_mb={size_mb:.1f} elapsed_sec={time.time()-started:.1f}")
    _audit_ane(output)


if __name__ == "__main__":
    main()
