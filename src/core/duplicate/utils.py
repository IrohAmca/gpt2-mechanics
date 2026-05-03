import re
from collections import defaultdict

import pandas as pd
import torch


def _as_input_ids(tokens):
    if isinstance(tokens, torch.Tensor):
        return tokens
    try:
        return tokens["input_ids"]
    except (KeyError, TypeError):
        pass
    raise TypeError("tokens must be a tensor or a mapping containing 'input_ids'.")


def _as_attention_mask(tokens):
    try:
        return tokens.get("attention_mask")
    except AttributeError:
        return None
    return None


def _model_device(model):
    cfg_device = getattr(getattr(model, "cfg", None), "device", None)
    if cfg_device is not None:
        return cfg_device
    return next(model.parameters()).device


def _ensure_batched(tokens):
    return tokens.unsqueeze(0) if tokens.ndim == 1 else tokens


def _metadata_list(metadata):
    if isinstance(metadata, dict):
        return [metadata]
    return list(metadata)


def _stack_cache_values(values):
    values = [value[0] if isinstance(value, (list, tuple)) else value for value in values]

    if values[0].ndim == 4 and values[0].shape[0] == 1:
        return torch.cat(values, dim=0)

    return torch.stack(values, dim=0)


def _iter_cache_items(cache):
    if hasattr(cache, "cache_dict"):
        return cache.cache_dict.items()

    if isinstance(cache, dict):
        return cache.items()

    if isinstance(cache, (list, tuple)) and cache and isinstance(cache[0], dict):
        return (
            (block_name, _stack_cache_values([sample_cache[block_name] for sample_cache in cache]))
            for block_name in cache[0]
        )

    return enumerate(cache)


def _is_pattern_name(block_name):
    return isinstance(block_name, int) or str(block_name).endswith("attn.hook_pattern")


def _prepare_attention_pattern(pattern):
    if isinstance(pattern, (list, tuple)):
        pattern = pattern[0]

    if pattern.ndim == 3:
        return pattern
    if pattern.ndim == 4:
        return pattern

    raise ValueError(
        "Attention pattern must have shape [heads, seq, seq] or "
        f"[batch, heads, seq, seq], got {tuple(pattern.shape)}."
    )


def _layer_index_from_block_name(block_name):
    if isinstance(block_name, int):
        return block_name

    match = re.search(r"blocks\.(\d+)", str(block_name))
    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot infer layer index from block name: {block_name!r}.")


def _metadata_index_tensor(metadata, key, device):
    values = []
    for row in _metadata_list(metadata):
        value = row.get(key)
        if value is None:
            span_key = key.replace("_index", "_span")
            span = row.get(span_key)
            if span is None or len(span) != 1:
                raise ValueError(
                    f"metadata entry needs a single-token {key!r} or {span_key!r}."
                )
            value = span[0]
        values.append(int(value))
    return torch.tensor(values, dtype=torch.long, device=device)


def _metadata_token_tensor(metadata, key, tokenizer=None, text_key=None, device=None):
    values = []
    for row in _metadata_list(metadata):
        value = row.get(key)
        if value is None:
            if tokenizer is None or text_key is None:
                raise ValueError(f"metadata entry is missing {key!r}.")
            token_ids = tokenizer.encode(" " + row[text_key], add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Expected {row[text_key]!r} to be one token.")
            value = token_ids[0]
        values.append(int(value))
    return torch.tensor(values, dtype=torch.long, device=device)


def _prediction_positions(tokens, metadata, logits_device):
    attention_mask = _as_attention_mask(tokens)
    if attention_mask is not None:
        attention_mask = _ensure_batched(attention_mask).to(logits_device)
        return attention_mask.sum(dim=-1).long() - 1

    rows = _metadata_list(metadata)
    if all("prediction_token_index" in row for row in rows):
        return torch.tensor(
            [int(row["prediction_token_index"]) for row in rows],
            dtype=torch.long,
            device=logits_device,
        )

    input_ids = _ensure_batched(_as_input_ids(tokens))
    return torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1] - 1,
        dtype=torch.long,
        device=logits_device,
    )


def validate_duplicate_token_alignment(metadata, raise_on_error=True):
    """Check that duplicate-token positions are fixed across the dataset."""
    rows = _metadata_list(metadata)

    def unique_values(key):
        return sorted(
            {row.get(key) for row in rows},
            key=lambda value: -1 if value is None else value,
        )

    checks = {
        "first_repeat_token_index": unique_values("first_repeat_token_index"),
        "second_repeat_token_index": unique_values("second_repeat_token_index"),
        "prediction_token_index": unique_values("prediction_token_index"),
    }
    checks["is_aligned"] = all(
        len(values) == 1 and values[0] is not None
        for values in checks.values()
    )

    if raise_on_error and not checks["is_aligned"]:
        raise ValueError(f"Duplicate-token metadata is not aligned: {checks}")

    return checks


def calculate_duplicate_attention_scores(cache, metadata):
    """Score each head by attention from the second A token back to the first A."""
    scores = {}

    for block_name, pattern in _iter_cache_items(cache):
        if not _is_pattern_name(block_name):
            continue

        pattern = _prepare_attention_pattern(pattern)
        query_positions = _metadata_index_tensor(
            metadata, "second_repeat_token_index", pattern.device
        )
        key_positions = _metadata_index_tensor(
            metadata, "first_repeat_token_index", pattern.device
        )

        if pattern.ndim == 3:
            validate_duplicate_token_alignment(metadata)
            query_pos = int(query_positions[0].item())
            key_pos = int(key_positions[0].item())
            scores[block_name] = pattern[:, query_pos, key_pos].detach().cpu()
            continue

        batch_idx = torch.arange(pattern.shape[0], device=pattern.device)
        scores[block_name] = pattern[
            batch_idx, :, query_positions, key_positions
        ].mean(dim=0).detach().cpu()

    if not scores:
        raise ValueError("No attention pattern entries were found in the cache.")

    return scores


def duplicate_scores_to_dataframe(duplicate_scores):
    """Convert duplicate attention scores to a tidy layer/head table."""
    rows = []
    for block_name, head_scores in duplicate_scores.items():
        layer = _layer_index_from_block_name(block_name)
        for head, score in enumerate(head_scores.detach().cpu().tolist()):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "block": block_name,
                    "duplicate_attention_score": float(score),
                }
            )

    return pd.DataFrame(rows).sort_values(["layer", "head"]).reset_index(drop=True)


def rank_duplicate_heads(duplicate_scores, top_k=20):
    """Return the most active duplicate-token heads."""
    df = duplicate_scores_to_dataframe(duplicate_scores)
    return (
        df.sort_values("duplicate_attention_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def find_top_duplicate_heads(cache, metadata, top_k=20):
    """Calculate duplicate scores and return the top layer/head pairs."""
    scores = calculate_duplicate_attention_scores(cache, metadata)
    return rank_duplicate_heads(scores, top_k=top_k)


def duplicate_logit_diff_metric(
    logits,
    tokens,
    metadata,
    tokenizer=None,
    return_per_sample=False,
):
    """Mean logit(correct B) - logit(duplicated A) at the prompt end."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    rows = _metadata_list(metadata)
    device = logits.device
    positions = _prediction_positions(tokens, rows, device)
    correct_token_ids = _metadata_token_tensor(
        rows,
        "correct_token_id",
        tokenizer=tokenizer,
        text_key="correct_name",
        device=device,
    )
    incorrect_token_ids = _metadata_token_tensor(
        rows,
        "incorrect_token_id",
        tokenizer=tokenizer,
        text_key="incorrect_name",
        device=device,
    )
    batch_idx = torch.arange(logits.shape[0], device=device)
    per_sample = (
        logits[batch_idx, positions, correct_token_ids]
        - logits[batch_idx, positions, incorrect_token_ids]
    )

    if return_per_sample:
        return per_sample.detach().cpu()

    return per_sample.mean()


def _normalize_heads(heads=None, layers=None, n_heads=None):
    if heads is None:
        if layers is None:
            raise ValueError("Provide heads or layers to patch.")
        if n_heads is None:
            raise ValueError("n_heads is required when patching whole layers.")
        return [
            {"layer": int(layer), "head": int(head)}
            for layer in layers
            for head in range(n_heads)
        ]

    if isinstance(heads, pd.DataFrame):
        return [
            {"layer": int(row.layer), "head": int(row.head)}
            for row in heads.itertuples(index=False)
        ]

    normalized = []
    for item in heads:
        if isinstance(item, dict):
            normalized.append({"layer": int(item["layer"]), "head": int(item["head"])})
        else:
            layer, head = item
            normalized.append({"layer": int(layer), "head": int(head)})
    return normalized


def _group_heads_by_layer(heads):
    grouped = defaultdict(list)
    for item in heads:
        grouped[item["layer"]].append(item["head"])
    return {layer: sorted(set(heads)) for layer, heads in grouped.items()}


def _patch_positions(metadata, patch_positions, device):
    if patch_positions in (None, "all"):
        return None
    if patch_positions == "first_repeat":
        return _metadata_index_tensor(metadata, "first_repeat_token_index", device)
    if patch_positions == "second_repeat":
        return _metadata_index_tensor(metadata, "second_repeat_token_index", device)
    if patch_positions == "prediction":
        rows = _metadata_list(metadata)
        return torch.tensor(
            [int(row["prediction_token_index"]) for row in rows],
            dtype=torch.long,
            device=device,
        )
    if isinstance(patch_positions, torch.Tensor):
        return patch_positions.to(device)
    return torch.tensor(patch_positions, dtype=torch.long, device=device)


def _patch_head_z(corrupted_z, clean_z, heads, positions):
    clean_z = clean_z.to(corrupted_z.device)

    if corrupted_z.ndim == 3:
        if positions is None:
            corrupted_z[:, heads, :] = clean_z[:, heads, :]
        else:
            corrupted_z[positions, heads, :] = clean_z[positions, heads, :]
        return corrupted_z

    if corrupted_z.ndim != 4:
        raise ValueError(
            "hook_z must have shape [pos, heads, d_head] or "
            f"[batch, pos, heads, d_head], got {tuple(corrupted_z.shape)}."
        )

    if positions is None:
        corrupted_z[:, :, heads, :] = clean_z[:, :, heads, :]
        return corrupted_z

    batch_idx = torch.arange(corrupted_z.shape[0], device=corrupted_z.device)
    corrupted_z[batch_idx[:, None], positions[:, None], heads, :] = clean_z[
        batch_idx[:, None], positions[:, None], heads, :
    ]
    return corrupted_z


def patch_duplicate_heads(
    model,
    clean_tokens,
    corrupted_tokens,
    metadata,
    heads=None,
    layers=None,
    patch_positions="all",
    metric_fn=None,
    return_logits=False,
):
    """Patch selected attention head outputs from clean into corrupted prompts."""
    model_device = _model_device(model)
    clean_input_ids = _ensure_batched(_as_input_ids(clean_tokens)).to(model_device)
    corrupted_input_ids = _ensure_batched(_as_input_ids(corrupted_tokens)).to(
        model_device
    )
    selected_heads = _normalize_heads(
        heads=heads,
        layers=layers,
        n_heads=getattr(model.cfg, "n_heads", None),
    )
    heads_by_layer = _group_heads_by_layer(selected_heads)
    hook_names = {f"blocks.{layer}.attn.hook_z" for layer in heads_by_layer}

    if metric_fn is None:
        metric_fn = duplicate_logit_diff_metric

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(
            clean_input_ids,
            names_filter=lambda name: name in hook_names,
        )
        corrupted_logits = model(corrupted_input_ids)

        hooks = []
        for layer, layer_heads in heads_by_layer.items():
            hook_name = f"blocks.{layer}.attn.hook_z"

            def patch_hook(corrupted_z, hook, layer_heads=layer_heads):
                positions = _patch_positions(
                    metadata,
                    patch_positions,
                    corrupted_z.device,
                )
                return _patch_head_z(
                    corrupted_z,
                    clean_cache[hook.name],
                    layer_heads,
                    positions,
                )

            hooks.append((hook_name, patch_hook))

        patched_logits = model.run_with_hooks(corrupted_input_ids, fwd_hooks=hooks)

    clean_metric = metric_fn(clean_logits, clean_tokens, metadata)
    corrupted_metric = metric_fn(corrupted_logits, corrupted_tokens, metadata)
    patched_metric = metric_fn(patched_logits, corrupted_tokens, metadata)
    denominator = clean_metric - corrupted_metric
    recovery = (patched_metric - corrupted_metric) / denominator

    result = {
        "clean_metric": float(clean_metric.detach().cpu()),
        "corrupted_metric": float(corrupted_metric.detach().cpu()),
        "patched_metric": float(patched_metric.detach().cpu()),
        "recovery": float(recovery.detach().cpu()),
        "heads": selected_heads,
        "patch_positions": patch_positions,
    }
    if return_logits:
        result.update(
            {
                "clean_logits": clean_logits,
                "corrupted_logits": corrupted_logits,
                "patched_logits": patched_logits,
            }
        )

    return result
