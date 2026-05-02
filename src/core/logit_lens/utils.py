from __future__ import annotations

from collections.abc import Sequence

import torch
from transformer_lens import ActivationCache, HookedTransformer


BlockIds = int | Sequence[int]


def _as_block_ids(block_ids: BlockIds) -> list[int]:
    if isinstance(block_ids, int):
        return [block_ids]
    return list(block_ids)


def _get_tokenizer(model: HookedTransformer, tokenizer=None):
    tokenizer = tokenizer or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("A tokenizer is required for decoding token ids.")
    return tokenizer


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _ensure_block_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError("Expected logits with shape [block, batch, pos, d_vocab].")
    return logits


def _position_list(seq_len: int, positions: int | Sequence[int] | None) -> list[int]:
    if positions is None:
        return list(range(seq_len))
    if isinstance(positions, int):
        return [positions % seq_len]
    return [pos % seq_len for pos in positions]


def get_block_residuals(
    block_ids: BlockIds,
    cache: ActivationCache,
) -> torch.Tensor:
    """Return cached block resid_post tensors as [block, batch, pos, d_model]."""

    blocks = _as_block_ids(block_ids)
    residuals = []

    for block_id in blocks:
        hook_name = f"blocks.{block_id}.hook_resid_post"
        if hook_name not in cache:
            raise ValueError(f"Block residual '{hook_name}' was not found in cache.")
        residuals.append(cache[hook_name])

    return torch.stack(residuals, dim=0)


def get_block_logits(
    model: HookedTransformer,
    block_ids: BlockIds,
    cache: ActivationCache,
    tokenizer=None,
    return_decoded: bool = False,
    apply_ln: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, list[list[list[str]]]]:
    """Project block residual streams through the unembed matrix.

    Returns logits with shape [block, batch, pos, d_vocab]. If return_decoded=True,
    also returns argmax token strings as [block][batch][pos].
    """

    residuals = get_block_residuals(block_ids, cache)

    if apply_ln:
        residuals = cache.apply_ln_to_stack(
            residuals,
            layer=model.cfg.n_layers,
            recompute_ln=True,
        )

    logits = model.unembed(residuals)

    if not return_decoded:
        return logits

    decoded = decode_argmax_tokens(model, logits, tokenizer=tokenizer)
    return logits, decoded


def decode_argmax_tokens(
    model: HookedTransformer,
    logits: torch.Tensor,
    tokenizer=None,
) -> list[list[list[str]]]:
    """Decode argmax predictions as [block][batch][pos] token strings."""

    tokenizer = _get_tokenizer(model, tokenizer)
    token_ids = logits.argmax(dim=-1).detach().cpu()

    if token_ids.ndim == 2:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.ndim != 3:
        raise ValueError(
            "Expected logits with shape [block, batch, pos, d_vocab] "
            "or [batch, pos, d_vocab]."
        )

    return [
        [
            [_decode_token(tokenizer, token_id) for token_id in batch_token_ids]
            for batch_token_ids in block_token_ids
        ]
        for block_token_ids in token_ids
    ]
