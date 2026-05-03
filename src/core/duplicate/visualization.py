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


def _ensure_batched(tokens):
    return tokens.unsqueeze(0) if tokens.ndim == 1 else tokens


def duplicate_attention_for_circuitsvis(
    cache,
    tokens,
    tokenizer,
    layer,
    batch_idx=0,
    heads=None,
):
    """Return token labels and a layer attention tensor ready for circuitsvis."""
    try:
        pattern = cache["pattern", layer, "attn"]
    except Exception:
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

    if isinstance(pattern, (list, tuple)):
        pattern = pattern[0]
    if pattern.ndim == 4:
        pattern = pattern[batch_idx]

    if heads is not None:
        pattern = pattern[heads]

    input_ids = _ensure_batched(_as_input_ids(tokens))[batch_idx].detach().cpu().tolist()
    attention_mask = _as_attention_mask(tokens)
    if attention_mask is not None:
        seq_len = int(_ensure_batched(attention_mask)[batch_idx].sum().item())
        input_ids = input_ids[:seq_len]
        pattern = pattern[..., :seq_len, :seq_len]

    token_labels = [tokenizer.decode([token_id]) for token_id in input_ids]
    return {
        "tokens": token_labels,
        "attention": pattern.detach().cpu(),
    }


def show_duplicate_attention(
    cache,
    tokens,
    tokenizer,
    layer,
    batch_idx=0,
    heads=None,
):
    """Render a duplicate-attention pattern with circuitsvis."""
    import circuitsvis as cv

    data = duplicate_attention_for_circuitsvis(
        cache=cache,
        tokens=tokens,
        tokenizer=tokenizer,
        layer=layer,
        batch_idx=batch_idx,
        heads=heads,
    )

    try:
        return cv.attention.attention_heads(
            tokens=data["tokens"],
            attention=data["attention"],
        )
    except AttributeError:
        return cv.attention.attention_patterns(
            tokens=data["tokens"],
            attention=data["attention"],
        )
