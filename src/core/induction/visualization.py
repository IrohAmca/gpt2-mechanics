import re

import pandas as pd
import torch


def _block_to_layer(block_name):
    match = re.search(r"blocks\.(\d+)", str(block_name))
    if match:
        return int(match.group(1))

    if isinstance(block_name, int):
        return block_name

    return block_name


def _to_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()

    return list(values)


def scores_to_dataframe(induction_scores, logit_attribution_scores=None):
    """Convert score dictionaries into one tidy DataFrame.

    Output columns:
        layer, head, block, previous_token_score, induction_score,
        logit_attribution_score
    """
    rows = []

    for block_name, block_scores in induction_scores.items():
        previous_scores = _to_list(block_scores["previous_token_score"])
        induction_head_scores = _to_list(block_scores["induction_score"])

        for head_idx, (previous_score, induction_score) in enumerate(
            zip(previous_scores, induction_head_scores)
        ):
            row = {
                "layer": _block_to_layer(block_name),
                "head": head_idx,
                "block": block_name,
                "previous_token_score": float(previous_score),
                "induction_score": float(induction_score),
            }

            if logit_attribution_scores is not None:
                z_block_name = str(block_name).replace("hook_pattern", "hook_z")
                attribution_values = logit_attribution_scores.get(
                    z_block_name, logit_attribution_scores.get(block_name)
                )
                if attribution_values is not None:
                    row["logit_attribution_score"] = float(
                        _to_list(attribution_values)[head_idx]
                    )

            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["layer", "head"]).reset_index(drop=True)
    if "logit_attribution_score" in df.columns:
        df["logit_attribution_score"] = pd.to_numeric(
            df["logit_attribution_score"], errors="coerce"
        )

    return df


def rank_induction_heads(
    induction_scores,
    logit_attribution_scores=None,
    sort_by="induction_score",
    top_k=20,
):
    """Return the highest-scoring heads as a DataFrame."""
    df = scores_to_dataframe(induction_scores, logit_attribution_scores)
    return df.sort_values(sort_by, ascending=False).head(top_k).reset_index(drop=True)


def score_grid(induction_scores, metric="induction_score"):
    """Return a layer x head grid for one metric."""
    df = scores_to_dataframe(induction_scores)
    return df.pivot(index="layer", columns="head", values=metric)


def display_induction_scores(
    induction_scores,
    logit_attribution_scores=None,
    sort_by="induction_score",
    top_k=20,
):
    """Display a compact ranked table in notebooks and return the DataFrame."""
    df = rank_induction_heads(
        induction_scores,
        logit_attribution_scores=logit_attribution_scores,
        sort_by=sort_by,
        top_k=top_k,
    )

    display_columns = [
        "layer",
        "head",
        "previous_token_score",
        "induction_score",
    ]
    if "logit_attribution_score" in df.columns:
        display_columns.append("logit_attribution_score")

    styled = df[display_columns].style.format(
        {
            "previous_token_score": "{:.3f}",
            "induction_score": "{:.3f}",
            "logit_attribution_score": "{:.3f}",
        },
        na_rep="nan",
    )

    try:
        from IPython.display import display

        display(styled)
    except ImportError:
        print(df[display_columns].to_string(index=False))

    return df
