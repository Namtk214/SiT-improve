"""Helper utilities for common/private activation decomposition in DiT."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

DEFAULT_SPATIAL_WINDOW_SIZE = 3


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


def tokens_to_grid(x: jax.Array) -> jax.Array:
    """Reshape `[B, N, C]` tokens into a square spatial grid `[B, H, W, C]`."""
    if x.ndim != 3:
        raise ValueError(f"Expected tokens with rank 3, got shape {x.shape}")
    batch, num_tokens, channels = x.shape
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Expected a square token grid, got N={num_tokens}")
    return x.reshape(batch, grid_size, grid_size, channels)


def _normalize_channels(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def extract_sliding_windows(
    grid: jax.Array,
    window_size: int,
) -> jax.Array:
    """Extract all valid sliding windows as `[B, num_windows, window_area, C]`."""
    if grid.ndim != 4:
        raise ValueError(f"Expected a spatial grid with rank 4, got shape {grid.shape}")
    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")

    batch, height, width, channels = grid.shape
    if window_size > height or window_size > width:
        raise ValueError(
            f"Window size {window_size} exceeds grid size {(height, width)}"
        )

    out_h = height - window_size + 1
    out_w = width - window_size + 1
    window_tokens = []
    for dy in range(window_size):
        for dx in range(window_size):
            window_tokens.append(grid[:, dy:dy + out_h, dx:dx + out_w, :])

    stacked = jnp.stack(window_tokens, axis=3)  # [B, out_h, out_w, window_area, C]
    return stacked.reshape(batch, out_h * out_w, window_size * window_size, channels)


def window_gram_matrix(
    window_tokens: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute normalized local Gram matrices for `[B, W, M, C]` window tokens."""
    if window_tokens.ndim != 4:
        raise ValueError(
            f"Expected window tokens with rank 4, got shape {window_tokens.shape}"
        )
    normalized = _normalize_channels(window_tokens, eps=eps)
    return jnp.einsum("bwmc,bwnc->bwmn", normalized, normalized)


def local_window_gram_loss(
    feature_tokens: jax.Array,
    target_tokens: jax.Array,
    window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
    eps: float = 1e-8,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compare local Gram structure between feature and target sliding windows."""
    feature_grid = tokens_to_grid(feature_tokens)
    target_grid = tokens_to_grid(target_tokens)
    if feature_grid.shape[:3] != target_grid.shape[:3]:
        raise ValueError(
            "Feature and target grids must match in batch/height/width, "
            f"got {feature_grid.shape[:3]} vs {target_grid.shape[:3]}"
        )

    feature_windows = extract_sliding_windows(feature_grid, window_size)
    target_windows = extract_sliding_windows(target_grid, window_size)
    feature_grams = window_gram_matrix(feature_windows, eps=eps)
    target_grams = window_gram_matrix(target_windows, eps=eps)

    window_losses = jnp.mean(jnp.abs(feature_grams - target_grams), axis=(-2, -1))
    spatial_loss = jnp.mean(window_losses)
    spatial_metrics = {
        "spatial_num_windows": jnp.asarray(feature_windows.shape[1], dtype=feature_tokens.dtype),
        "spatial_window_area": jnp.asarray(feature_windows.shape[2], dtype=feature_tokens.dtype),
    }
    return spatial_loss, spatial_metrics


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
    spatial_window_size: int = DEFAULT_SPATIAL_WINDOW_SIZE,
) -> dict[str, jax.Array]:
    """Compute auxiliary losses and logging metrics for activation decomposition."""
    activations = collect_activations(activations)
    if spatial_target.ndim != 3:
        raise ValueError(f"Expected spatial target with rank 3, got shape {spatial_target.shape}")

    common, common_anchor, private = compute_common_private(activations)
    common_loss = jnp.mean(jnp.square(activations - common_anchor[None, ...]))

    spatial_loss, spatial_metrics = local_window_gram_loss(
        common,
        spatial_target,
        window_size=spatial_window_size,
    )

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
        "spatial_metrics": spatial_metrics,
        "norm_common": common_norm,
        "avg_private_norm": avg_private_norm,
        "avg_pairwise_private_cosine": avg_pairwise_cosine,
    }
