import torch


def _get_bos_token_id(tokenizer):
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id for induction data.")
    return bos_token_id


def _special_token_ids(tokenizer):
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        special_ids.add(bos_token_id)
    return special_ids


def _sample_token_ids(tokenizer, n, generator=None, exclude_special=True):
    vocab_size = len(tokenizer)
    if vocab_size <= 0:
        raise ValueError("Tokenizer vocabulary is empty.")

    special_ids = _special_token_ids(tokenizer) if exclude_special else set()
    sampled = []
    while len(sampled) < n:
        candidate = torch.randint(
            low=0,
            high=vocab_size,
            size=(1,),
            generator=generator,
        ).item()
        if candidate not in special_ids:
            sampled.append(candidate)
    return sampled


def _resolve_lengths(seq_len, prefix_len, pattern_len):
    if seq_len < 6:
        raise ValueError(
            "seq_len must be at least 6: BOS + prefix + two copies of a 2-token pattern."
        )

    available = seq_len - 1
    if prefix_len is None and pattern_len is None:
        pattern_len = max(2, available // 3)
        prefix_len = available - 2 * pattern_len
    elif prefix_len is None:
        prefix_len = available - 2 * pattern_len
    elif pattern_len is None:
        remaining = available - prefix_len
        if remaining % 2 != 0:
            raise ValueError(
                "seq_len - 1 - prefix_len must be even when pattern_len is omitted."
            )
        pattern_len = remaining // 2

    if prefix_len < 1:
        raise ValueError("prefix_len must leave at least one random prefix token.")
    if pattern_len < 2:
        raise ValueError("pattern_len must be at least 2 for an induction signal.")
    if 1 + prefix_len + 2 * pattern_len != seq_len:
        raise ValueError(
            "Lengths must satisfy seq_len == 1 + prefix_len + 2 * pattern_len."
        )

    return prefix_len, pattern_len


def generate_induction_data(
    tokenizer,
    num_samples=1000,
    seq_len=32,
    prefix_len=None,
    pattern_len=None,
    device=None,
    generator=None,
    exclude_special=True,
    return_metadata=False,
):
    """Generate synthetic induction-head data.

    Each sequence has this structure:
        [BOS] + random_prefix + repeated_pattern + repeated_pattern

    The random prefix prevents the model from solving the task from absolute
    positions alone. The second copy of the pattern creates the induction
    signal: after seeing token x in the second copy, the model can attend to
    the previous x and predict the token that followed it in the first copy.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    bos_token_id = _get_bos_token_id(tokenizer)
    prefix_len, pattern_len = _resolve_lengths(seq_len, prefix_len, pattern_len)

    data = []
    for _ in range(num_samples):
        prefix = _sample_token_ids(
            tokenizer,
            prefix_len,
            generator=generator,
            exclude_special=exclude_special,
        )
        pattern = _sample_token_ids(
            tokenizer,
            pattern_len,
            generator=generator,
            exclude_special=exclude_special,
        )
        data.append([bos_token_id, *prefix, *pattern, *pattern])

    tokens = torch.tensor(data, dtype=torch.long, device=device)
    if not return_metadata:
        return tokens

    first_pattern_start = 1 + prefix_len
    second_pattern_start = first_pattern_start + pattern_len
    metadata = {
        "prefix_start": 1,
        "prefix_end": first_pattern_start,
        "first_pattern_start": first_pattern_start,
        "first_pattern_end": second_pattern_start,
        "second_pattern_start": second_pattern_start,
        "second_pattern_end": seq_len,
        "induction_source_positions": torch.arange(
            first_pattern_start,
            second_pattern_start - 1,
            dtype=torch.long,
            device=device,
        ),
        "induction_query_positions": torch.arange(
            second_pattern_start,
            seq_len - 1,
            dtype=torch.long,
            device=device,
        ),
    }
    return tokens, metadata
