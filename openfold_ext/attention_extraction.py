# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Extracts self-attention weight matrices from an OpenFold3 inference run.

No layer in the model returns its attention weights: the softmax is computed
and immediately consumed inside the free function
`openfold3.core.model.primitives.attention._attention`. This script
monkey-patches that single function (shared by every attention layer in the
network) to also capture the post-softmax weights, tagged with the qualified
module name of whichever `Attention` instance is currently calling it (a
forward pre-hook maintains that name on a small stack around each call).

Only three layer families are captured, matching the ones exposed in the
Django UI:
  - "pairformer_self_attn": PairFormerBlock.attn_pair_bias (residue-to-residue,
    48 blocks x 16 heads)
  - "triangle_start" / "triangle_end": tri_att_start / tri_att_end, present in
    the Pairformer, MSAModule and TemplatePairStack (4 heads)
  - "diffusion_token": DiffusionTransformerBlock.attention_pair_bias, captured
    at a single representative denoising step (the full ~200-step rollout
    would be far too much data to keep)

Weights are averaged over heads and stored as float16 .npz arrays per family,
plus a meta.json with per-layer entropy (to distinguish local vs global
attention patterns) and run metadata.
"""

import contextlib
import json
import logging
import math
import threading
from pathlib import Path

import click
import numpy as np
import torch

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import _torch_gpu_setup

logger = logging.getLogger(__name__)

# Family name -> substring(s) that must all appear in a module's qualified name
# for it to be captured. Matched against names produced by nn.Module.named_modules().
#
# Triangle attention (tri_att_start/tri_att_end) is a shared building block
# (base_blocks.py::PairBlock) reused by four different stacks: the main
# Pairformer trunk, the MSAModule, the TemplatePairStack, and a *second*,
# separate Pairformer instance nested inside the confidence heads
# (aux_heads.pairformer_embedding.pairformer_stack). Only the main trunk
# ("pairformer_stack.blocks.N...", with nothing before "pairformer_stack") is
# captured here, so a family's blocks are always a single homogeneous stack.
#
# DiffusionModule similarly reuses AttentionPairBias/CrossAttentionPairBias in
# three places with different tensor shapes: diffusion_transformer (token
# level, [*, H, N_token, N_token]) vs atom_attn_enc/atom_attn_dec (windowed
# atom-level, [*, n_blocks, H, n_query, n_key]). Only diffusion_transformer is
# captured under "diffusion_token" so every captured array has the same shape.
LAYER_FAMILIES = {
    "pairformer_self_attn": ("pairformer_stack.blocks.", "attn_pair_bias.mha"),
    "triangle_start": ("pairformer_stack.blocks.", "tri_att_start.mha"),
    "triangle_end": ("pairformer_stack.blocks.", "tri_att_end.mha"),
    "diffusion_token": ("diffusion_module.diffusion_transformer.blocks.", "attention_pair_bias.mha"),
}


def _family_for_name(qualified_name: str) -> str | None:
    if not qualified_name.startswith(
        "pairformer_stack."
    ) and not qualified_name.startswith("diffusion_module.diffusion_transformer."):
        return None
    for family, needles in LAYER_FAMILIES.items():
        if all(needle in qualified_name for needle in needles):
            return family
    return None


class AttentionCapture:
    """Owns the monkey-patch of `_attention` and the resulting captured maps.

    Captured maps are keyed by qualified module name and stored as a running
    list (the diffusion token attention layers are called once per denoising
    step, so we keep every call and let the caller pick a representative
    step afterwards).
    """

    def __init__(self, families: set[str]):
        self.families = families
        self.captures: dict[str, list[np.ndarray]] = {}
        self._context = threading.local()
        self._orig_attention = None

    def _current_layer_name(self) -> str | None:
        stack = getattr(self._context, "stack", None)
        if not stack:
            return None
        return stack[-1]

    def _make_pre_hook(self, qualified_name: str):
        def _pre_hook(module, args, kwargs):
            stack = getattr(self._context, "stack", None)
            if stack is None:
                stack = []
                self._context.stack = stack
            stack.append(qualified_name)

        return _pre_hook

    def _make_post_hook(self):
        def _post_hook(module, args, kwargs, output):
            stack = getattr(self._context, "stack", None)
            if stack:
                stack.pop()

        return _post_hook

    def register(self, model: torch.nn.Module) -> int:
        """Registers pre/post hooks on every Attention submodule that
        belongs to one of the requested families. Returns the hook count."""
        count = 0
        for name, module in model.named_modules():
            family = _family_for_name(name)
            if family is None or family not in self.families:
                continue
            if type(module).__name__ != "Attention":
                continue
            module.register_forward_pre_hook(
                self._make_pre_hook(name), with_kwargs=True
            )
            module.register_forward_hook(self._make_post_hook(), with_kwargs=True)
            count += 1
        return count

    @contextlib.contextmanager
    def patch(self):
        """Monkey-patches `_attention` for the duration of the `with` block.

        Delegates to the original function for the actual computation (so the
        model's output is guaranteed byte-for-byte identical to an unpatched
        run) and only recomputes the softmax weights separately, purely for
        capture. The extra softmax is cheap relative to the rest of a trunk
        pass and keeps this a pure "spy" with zero risk of numerical drift.
        """
        import openfold3.core.model.primitives.attention as attention_module

        self._orig_attention = attention_module._attention
        orig_attention = self._orig_attention
        capture_self = self

        def _spying_attention(query, key, value, biases, use_high_precision=False):
            layer_name = capture_self._current_layer_name()
            if layer_name is not None:
                with torch.no_grad():
                    attn_dtype = torch.float32 if use_high_precision else query.dtype
                    with torch.amp.autocast("cuda", dtype=attn_dtype, enabled=False):
                        scores = torch.einsum("...qc, ...kc->...qk", query, key)
                        for b in biases:
                            scores = scores + b
                        scores = torch.nn.functional.softmax(scores, dim=-1)
                    # scores: [*, H, Q, K] -> average over heads, keep batch dims
                    weights = scores.mean(dim=-3).to(torch.float16).cpu().numpy()
                capture_self.captures.setdefault(layer_name, []).append(weights)

            return orig_attention(
                query, key, value, biases, use_high_precision=use_high_precision
            )

        attention_module._attention = _spying_attention
        try:
            yield
        finally:
            attention_module._attention = self._orig_attention


def _entropy(weights: np.ndarray) -> float:
    """Mean per-row Shannon entropy of an [N, N] (or [*, N, N]) attention map."""
    w = weights.astype(np.float64)
    w = np.clip(w, 1e-12, 1.0)
    row_entropy = -(w * np.log(w)).sum(axis=-1)
    return float(row_entropy.mean())


TRIANGLE_FAMILIES = {"triangle_start", "triangle_end"}


def _squeeze_leading(arr: np.ndarray, keep_dims: int) -> np.ndarray:
    """Drops leading singleton dims until `keep_dims` dims remain.

    The trunk has one batch dim; the diffusion rollout has an extra
    no_rollout_samples dim on top of that.
    """
    while arr.ndim > keep_dims and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _save_family(
    output_dir: Path,
    family: str,
    captures: dict[str, list[np.ndarray]],
) -> dict:
    """Saves all captured layers for one family into a single .npz, sorted by
    block index (parsed from the qualified module name). Returns metadata.

    Triangle attention is not a plain residue-to-residue map: for each pair
    (i, j) of the pair representation it attends over a third index k, so the
    captured tensor is [I, J, K] rather than [N, N] (TriangleAttention treats
    I as an extra batch dimension over the pair representation — see
    triangular_attention.py::TriangleAttention.forward). Both a full [I,J,K]
    cube and its k-averaged [I,J] heatmap are saved for these families so the
    UI can offer either view.
    """
    layer_names = [
        name for name in captures if _family_for_name(name) == family
    ]

    def _block_index(name: str) -> int:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "blocks" and i + 1 < len(parts):
                return int(parts[i + 1])
        return 0

    layer_names.sort(key=_block_index)

    if not layer_names:
        return {"blocks": 0, "layer_names": [], "is_triangle": False}

    is_triangle = family in TRIANGLE_FAMILIES
    keep_dims = 3 if is_triangle else 2

    if family == "diffusion_token":
        # Called once per denoising step; keep only the middle step as a
        # representative snapshot (the full rollout is too much to store).
        arrays = [captures[name][len(captures[name]) // 2] for name in layer_names]
    else:
        # Called once per forward pass (single trunk pass), one array per block.
        arrays = [captures[name][0] for name in layer_names]

    arrays = [_squeeze_leading(arr, keep_dims) for arr in arrays]

    if is_triangle:
        cube_stacked = np.stack(arrays, axis=0)  # [n_blocks, I, J, K]
        np.savez_compressed(output_dir / f"{family}_cube.npz", weights=cube_stacked)

        heatmaps = [arr.mean(axis=-1) for arr in arrays]  # [I, J], averaged over k
        stacked = np.stack(heatmaps, axis=0)
        np.savez_compressed(output_dir / f"{family}.npz", weights=stacked)

        entropy_per_block = [_entropy(arr) for arr in heatmaps]

        return {
            "blocks": stacked.shape[0],
            "n_tokens": stacked.shape[-1],
            "layer_names": layer_names,
            "entropy_per_block": entropy_per_block,
            "is_triangle": True,
            "cube_shape": list(cube_stacked.shape),
        }

    stacked = np.stack(arrays, axis=0)  # [n_blocks, N, N]
    np.savez_compressed(output_dir / f"{family}.npz", weights=stacked)

    entropy_per_block = [_entropy(arr) for arr in arrays]

    return {
        "blocks": stacked.shape[0],
        "n_tokens": stacked.shape[-1],
        "layer_names": layer_names,
        "entropy_per_block": entropy_per_block,
        "is_triangle": False,
    }


@click.command()
@click.option(
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--families",
    type=str,
    default="pairformer_self_attn,triangle_start,triangle_end,diffusion_token",
    help="Comma-separated list of layer families to capture.",
)
def extract(
    query_json: Path,
    runner_yaml: Path | None,
    output_dir: Path,
    families: str,
):
    """Runs a single-sample OpenFold3 inference pass with attention capture."""
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    logging.basicConfig(level=logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_families = set(f.strip() for f in families.split(",") if f.strip())
    unknown = requested_families - set(LAYER_FAMILIES)
    if unknown:
        raise click.BadParameter(f"Unknown families: {sorted(unknown)}")

    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()
    expt_config = InferenceExperimentConfig(**runner_args)
    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=1,
        num_model_seeds=1,
        use_msa_server=True,
        use_templates=True,
        output_dir=output_dir,
    )

    query_set = InferenceQuerySet.from_json(query_json)

    expt_runner.setup()

    capture = AttentionCapture(requested_families)
    hook_count = capture.register(expt_runner.lightning_module.model)
    logger.info(f"Registered {hook_count} attention hooks for families: {requested_families}")

    with capture.patch():
        expt_runner.run(query_set)

    meta = {"families": {}, "requested_families": sorted(requested_families)}
    for family in requested_families:
        meta["families"][family] = _save_family(output_dir, family, capture.captures)

    sequence = next(iter(query_set.queries.values())).chains[0].sequence
    meta["sequence"] = sequence
    meta["n_residues"] = len(sequence)

    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Attention extraction complete. Wrote meta.json to {output_dir}")

    expt_runner.cleanup()


if __name__ == "__main__":
    extract()
