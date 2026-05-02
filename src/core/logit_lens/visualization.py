from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from core.logit_lens.utils import (
    BlockIds,
    _as_block_ids,
    _decode_token,
    _ensure_block_logits,
    _get_tokenizer,
    _position_list,
)


def _format_token(token: str) -> str:
    if token == "\n":
        return "\\n"
    return token.replace("\n", "\\n")


def top_token_table(
    model: HookedTransformer,
    logits: torch.Tensor,
    block_ids: BlockIds,
    tokenizer=None,
    input_ids: torch.Tensor | None = None,
    batch_idx: int = 0,
    positions: int | Sequence[int] | None = -1,
    top_k: int = 5,
) -> pd.DataFrame:
    """Build a tidy table of top-k logit-lens predictions."""

    tokenizer = _get_tokenizer(model, tokenizer)
    blocks = _as_block_ids(block_ids)

    logits = _ensure_block_logits(logits)
    if len(blocks) != logits.shape[0]:
        raise ValueError("block_ids length must match the first logits dimension.")

    pos_list = _position_list(logits.shape[2], positions)

    rows = []
    for block_offset, block_id in enumerate(blocks):
        for pos in pos_list:
            values, token_ids = logits[block_offset, batch_idx, pos].topk(top_k)
            input_token = None
            if input_ids is not None:
                input_token = _decode_token(tokenizer, int(input_ids[batch_idx, pos]))

            for rank, (value, token_id) in enumerate(zip(values, token_ids), start=1):
                rows.append(
                    {
                        "block": block_id,
                        "pos": pos,
                        "input": input_token,
                        "rank": rank,
                        "token": _decode_token(tokenizer, int(token_id)),
                        "token_id": int(token_id),
                        "logit": float(value),
                    }
                )

    return pd.DataFrame(rows)


def argmax_token_table(
    model: HookedTransformer,
    logits: torch.Tensor,
    block_ids: BlockIds,
    tokenizer=None,
    input_ids: torch.Tensor | None = None,
    batch_idx: int = 0,
    positions: int | Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build a block-by-position table of argmax tokens."""

    tokenizer = _get_tokenizer(model, tokenizer)
    blocks = _as_block_ids(block_ids)
    logits = _ensure_block_logits(logits)

    if len(blocks) != logits.shape[0]:
        raise ValueError("block_ids length must match the first logits dimension.")

    pos_list = _position_list(logits.shape[2], positions)
    token_ids = logits[:, batch_idx].argmax(dim=-1).detach().cpu()

    columns = []
    for pos in pos_list:
        label = str(pos)
        if input_ids is not None:
            input_token = _format_token(_decode_token(tokenizer, int(input_ids[batch_idx, pos])))
            label = f"{pos}: {input_token}"
        columns.append(label)

    rows = []
    for block_offset, block_id in enumerate(blocks):
        row = {
            columns[col_idx]: _format_token(
                _decode_token(tokenizer, int(token_ids[block_offset, pos]))
            )
            for col_idx, pos in enumerate(pos_list)
        }
        row["block"] = block_id
        rows.append(row)

    return pd.DataFrame(rows).set_index("block")


def style_top_token_table(table: pd.DataFrame):
    """Return a notebook-friendly style for top_token_table output."""

    styled = table.copy()
    if "token" in styled:
        styled["token"] = styled["token"].map(_format_token)
    if "input" in styled:
        styled["input"] = styled["input"].map(
            lambda value: None if value is None else _format_token(value)
        )

    return (
        styled.style.hide(axis="index")
        .format({"logit": "{:.3f}"})
        .apply(_logit_background, subset=["logit"])
        .set_properties(
            subset=["token", "input"],
            **{
                "font-family": "monospace",
                "white-space": "pre",
            },
        )
    )


def _logit_background(values: pd.Series) -> list[str]:
    min_value = values.min()
    max_value = values.max()
    span = max(max_value - min_value, 1e-9)

    styles = []
    for value in values:
        alpha = 0.12 + 0.36 * ((value - min_value) / span)
        styles.append(f"background-color: rgba(38, 132, 255, {alpha:.3f})")
    return styles


def style_argmax_token_table(table: pd.DataFrame):
    """Return a notebook-friendly style for argmax_token_table output."""

    return table.style.set_properties(
        **{
            "font-family": "monospace",
            "white-space": "pre",
            "text-align": "left",
        }
    )


def display_top_tokens(
    model: HookedTransformer,
    logits: torch.Tensor,
    block_ids: BlockIds,
    tokenizer=None,
    input_ids: torch.Tensor | None = None,
    batch_idx: int = 0,
    positions: int | Sequence[int] | None = -1,
    top_k: int = 5,
) -> pd.DataFrame:
    """Display top-k predictions in notebooks and return the DataFrame."""

    from IPython.display import display

    table = top_token_table(
        model,
        logits,
        block_ids,
        tokenizer=tokenizer,
        input_ids=input_ids,
        batch_idx=batch_idx,
        positions=positions,
        top_k=top_k,
    )
    display(style_top_token_table(table))
    return table


def display_argmax_tokens(
    model: HookedTransformer,
    logits: torch.Tensor,
    block_ids: BlockIds,
    tokenizer=None,
    input_ids: torch.Tensor | None = None,
    batch_idx: int = 0,
    positions: int | Sequence[int] | None = None,
) -> pd.DataFrame:
    """Display block-by-position argmax tokens in notebooks and return the DataFrame."""

    from IPython.display import display

    table = argmax_token_table(
        model,
        logits,
        block_ids,
        tokenizer=tokenizer,
        input_ids=input_ids,
        batch_idx=batch_idx,
        positions=positions,
    )
    display(style_argmax_token_table(table))
    return table
