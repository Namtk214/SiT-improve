"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

LOCAL_SPATIAL_OFFSETS = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
SPATIAL_OFFSET_METRIC_NAMES = tuple(
    f"spatial_offset_dy{dy}_dx{dx}" for dy, dx in LOCAL_SPATIAL_OFFSETS
)


def collect_activations(activations: Any) -> jax.Array:
    """Normalize activations into a stacked `[L, B, N, D]` tensor."""
    if hasattr(activations, "ndim"):
        if activations.ndim != 4:
            raise ValueError(f"Expected activations with rank 4, got shape {activations.shape}")
        return activations
    if isinstance(activations, (list, tuple)):
        if not activations:
            raise ValueError("Expected at least one activation tensor.")
        stacked = jnp.stack(activations, axis=0)
        if stacked.ndim != 4:
            raise ValueError(f"Expected stacked activations with rank 4, got shape {stacked.shape}")
        return stacked
    raise TypeError(f"Unsupported activation container type: {type(activations)!r}")


def compute_common_private(activations: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute differentiable common activation and private residuals."""
    activations = collect_activations(activations)
    common = jnp.mean(activations, axis=0)
    common_anchor = jax.lax.stop_gradient(common)
    private = activations - common_anchor[None, ...]
    return common, common_anchor, private


def gram_matrix(x: jax.Array) -> jax.Array:
    """Compute batched Gram matrices from `[B, N, D]` to `[B, N, N]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected input with rank 3, got shape {x.shape}")
    return jnp.einsum("bnd,bmd->bnm", x, x)


def _offset_metric_name(dy: int, dx: int) -> str:
    return f"spatial_offset_dy{dy}_dx{dx}"


def tokens_to_grid(x: jax.Array) -> jax.Array:
    """Reshape `[B, N, C]` tokens into a square spatial grid `[B, H, W, C]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected tokens with rank 3, got shape {x.shape}")
    batch, num_tokens, channels = x.shape
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Expected a square token grid, got N={num_tokens}")
    return x.reshape(batch, grid_size, grid_size, channels)


def shifted_overlap_slices(height: int, width: int, dy: int, dx: int):
    """Return overlapping source/shifted slices for a 2D offset."""
    if abs(dy) >= height or abs(dx) >= width:
        return None

    if dy >= 0:
        src_h = slice(0, height - dy)
        dst_h = slice(dy, height)
    else:
        src_h = slice(-dy, height)
        dst_h = slice(0, height + dy)

    if dx >= 0:
        src_w = slice(0, width - dx)
        dst_w = slice(dx, width)
    else:
        src_w = slice(-dx, width)
        dst_w = slice(0, width + dx)

    return src_h, src_w, dst_h, dst_w


def _normalize_channels(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def local_cosine_similarity(
    grid: jax.Array,
    dy: int,
    dx: int,
    eps: float = 1e-8,
) -> jax.Array | None:
    """Compute cosine similarity map for one local offset."""
    if grid.ndim != 4:
        raise ValueError(f"Expected a spatial grid with rank 4, got shape {grid.shape}")

    overlap = shifted_overlap_slices(grid.shape[1], grid.shape[2], dy, dx)
    if overlap is None:
        return None

    src_h, src_w, dst_h, dst_w = overlap
    source = _normalize_channels(grid[:, src_h, src_w, :], eps=eps)
    shifted = _normalize_channels(grid[:, dst_h, dst_w, :], eps=eps)
    return jnp.sum(source * shifted, axis=-1)


def local_self_similarity_loss(
    feature_tokens: jax.Array,
    target_tokens: jax.Array,
    offsets: tuple[tuple[int, int], ...] = LOCAL_SPATIAL_OFFSETS,
    eps: float = 1e-8,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare local cosine self-similarity maps between feature and target tokens."""
    feature_grid = tokens_to_grid(feature_tokens)
    target_grid = tokens_to_grid(target_tokens)
    if feature_grid.shape[:3] != target_grid.shape[:3]:
        raise ValueError(
            "Feature and target grids must match in batch/height/width, "
            f"got {feature_grid.shape[:3]} vs {target_grid.shape[:3]}"
        )

    zero = jnp.array(0.0, dtype=feature_tokens.dtype)
    offset_losses = {name: zero for name in SPATIAL_OFFSET_METRIC_NAMES}
    valid_losses = []

    for dy, dx in offsets:
        feature_similarity = local_cosine_similarity(feature_grid, dy, dx, eps=eps)
        target_similarity = local_cosine_similarity(target_grid, dy, dx, eps=eps)
        if feature_similarity is None or target_similarity is None:
            continue

        offset_loss = jnp.mean(jnp.abs(feature_similarity - target_similarity))
        offset_losses[_offset_metric_name(dy, dx)] = offset_loss
        valid_losses.append(offset_loss)

    if not valid_losses:
        return zero, offset_losses

    return jnp.mean(jnp.stack(valid_losses)), offset_losses


def _pairwise_cosines(private: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Return cosine similarities for all layer pairs `i < j`."""
    num_layers = private.shape[0]
    if num_layers < 2:
        return jnp.array(0.0, dtype=private.dtype)

    flattened = private.reshape(num_layers, -1)
    norms = jnp.linalg.norm(flattened, axis=-1, keepdims=True)
    normalized = flattened / jnp.maximum(norms, eps)
    cosine_matrix = normalized @ normalized.T
    upper_indices = jnp.triu_indices(num_layers, k=1)
    return cosine_matrix[upper_indices]


def _mean_pairwise_cosine_squared(
    private: jax.Array,
    eps: float = 1e-8,
    rng: jax.Array | None = None,
    max_pairs: int = 0,
) -> jax.Array:
    """Average squared cosine similarity over all or sampled layer pairs."""
    pairwise_cosines = _pairwise_cosines(private, eps=eps)
    if pairwise_cosines.ndim == 0:
        return pairwise_cosines

    if max_pairs and max_pairs > 0 and pairwise_cosines.shape[0] > max_pairs:
        if rng is None:
            raise ValueError("An RNG key is required when sampling private-layer pairs.")
        indices = jax.random.permutation(rng, pairwise_cosines.shape[0])[:max_pairs]
        pairwise_cosines = pairwise_cosines[indices]

    return jnp.mean(jnp.square(pairwise_cosines))


def compute_aux_losses(
    activations: Any,
    spatial_target: jax.Array,
    private_pair_rng: jax.Array | None = None,
    private_max_pairs: int = 0,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses and logging metrics for activation decomposition."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")

    common, common_anchor, private = compute_common_private(activations)
    common_loss = jnp.mean(jnp.square(activations - common_anchor[None, ...]))

    spatial_loss, spatial_offset_losses = local_self_similarity_loss(common, spatial_target)

    private_loss = _mean_pairwise_cosine_squared(
        private,
        rng=private_pair_rng,
        max_pairs=private_max_pairs,
    )

    common_norm = jnp.mean(jnp.linalg.norm(common.reshape(common.shape[0], -1), axis=-1))
    private_norms = jnp.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), axis=-1)
    avg_private_norm = jnp.mean(private_norms)

    pairwise_cosines = _pairwise_cosines(private)
    if pairwise_cosines.ndim == 0:
        avg_pairwise_cosine = pairwise_cosines
    else:
        avg_pairwise_cosine = jnp.mean(pairwise_cosines)

    return {
        "common_activation": common,
        "private_activations": private,
        "loss_common": common_loss,
        "loss_spatial": spatial_loss,
        "loss_private": private_loss,
        "spatial_offset_losses": spatial_offset_losses,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
