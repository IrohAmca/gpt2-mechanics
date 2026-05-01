import torch


def _stack_cache_values(values):
    values = [value[0] if isinstance(value, (list, tuple)) else value for value in values]

    if values[0].ndim == 4 and values[0].shape[0] == 1:
        return torch.cat(values, dim=0)

    return torch.stack(values, dim=0)


def _iter_cache_items(cache):
    if isinstance(cache, dict):
        return cache.items()

    if isinstance(cache, (list, tuple)) and cache and isinstance(cache[0], dict):
        return (
            (block_name, _stack_cache_values([sample_cache[block_name] for sample_cache in cache]))
            for block_name in cache[0]
        )

    return enumerate(cache)


def _prepare_attention_pattern(pattern):
    if isinstance(pattern, (list, tuple)):
        pattern = pattern[0]

    if pattern.ndim == 4:
        pattern = pattern.mean(dim=0)
    elif pattern.ndim != 3:
        raise ValueError(
            "Attention pattern must have shape [heads, seq, seq] or "
            f"[batch, heads, seq, seq], got {tuple(pattern.shape)}."
        )

    return pattern


def _prepare_hook_z(z):
    if isinstance(z, (list, tuple)):
        z = z[0]

    if z.ndim == 3:
        z = z.unsqueeze(0)
    elif z.ndim != 4:
        raise ValueError(
            "hook_z must have shape [pos, heads, d_head] or "
            f"[batch, pos, heads, d_head], got {tuple(z.shape)}."
        )

    return z


def _layer_index_from_block_name(block_name):
    if isinstance(block_name, int):
        return block_name

    parts = str(block_name).split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        return int(parts[1])

    raise ValueError(f"Cannot infer layer index from block name: {block_name!r}.")


def _raise_if_nonfinite(name, tensor):
    if torch.isfinite(tensor).all():
        return

    nan_count = torch.isnan(tensor).sum().item()
    posinf_count = torch.isposinf(tensor).sum().item()
    neginf_count = torch.isneginf(tensor).sum().item()
    raise ValueError(
        f"{name} contains non-finite values "
        f"(nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}). "
        "Regenerate the cache with a numerically stable model dtype, usually float32."
    )


def _cal_previous_token_score(cache, metadata=None):
    """Calculate how strongly each attention head attends to the previous token."""
    scores = {}

    for block_name, pattern in _iter_cache_items(cache):
        pattern = _prepare_attention_pattern(pattern)
        scores[block_name] = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=-1)

    return scores


def _cal_induction_score(cache, metadata):
    """Calculate attention paid from the second pattern to next-token targets."""
    scores = {}

    for block_name, pattern in _iter_cache_items(cache):
        pattern = _prepare_attention_pattern(pattern)
        source_positions = metadata["induction_source_positions"].to(pattern.device)
        query_positions = metadata["induction_query_positions"].to(pattern.device)
        target_key_positions = source_positions + 1

        scores[block_name] = pattern[:, query_positions, target_key_positions].mean(
            dim=-1
        )

    return scores


def cal_induction_scores(cache, metadata):
    """Calculate previous-token and induction attention scores in one call."""
    previous_token_scores = _cal_previous_token_score(cache, metadata)
    induction_scores = _cal_induction_score(cache, metadata)

    return {
        block_name: {
            "previous_token_score": previous_token_scores[block_name],
            "induction_score": induction_scores[block_name],
        }
        for block_name in induction_scores
    }


def cal_logit_attribution_score(hook_z_cache, tokens, model, metadata):
    """Calculate each head's direct contribution to the correct next token."""
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    scores = {}
    compute_device = model.W_U.device
    tokens = tokens.to(compute_device)
    w_u = model.W_U.float()
    _raise_if_nonfinite("model.W_U", w_u)
    vocab_size = w_u.shape[-1]
    unembed_sum = w_u.sum(dim=-1)

    with torch.inference_mode():
        for block_name, z in _iter_cache_items(hook_z_cache):
            z = _prepare_hook_z(z).to(compute_device).float()
            _raise_if_nonfinite(f"{block_name}", z)
            layer_idx = _layer_index_from_block_name(block_name)

            query_positions = metadata["induction_query_positions"].to(compute_device)
            target_tokens = tokens[:, query_positions + 1].to(compute_device)

            z_at_queries = z[:, query_positions]
            w_o = model.W_O[layer_idx].float()
            _raise_if_nonfinite(f"model.W_O[{layer_idx}]", w_o)

            head_residual = torch.einsum("bphd,hdm->bphm", z_at_queries, w_o)
            target_unembed = w_u.T[target_tokens]
            logit_diff_direction = (
                vocab_size * target_unembed - unembed_sum
            ) / (vocab_size - 1)

            scores[block_name] = torch.einsum(
                "bphm,bpm->bph", head_residual, logit_diff_direction
            ).mean(dim=(0, 1)).detach().cpu()

    return scores
