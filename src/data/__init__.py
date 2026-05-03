from .duplicate_token_data import (
    CLEAN_PROMPTS,
    CORRUPTED_PROMPTS,
    DUPLICATE_TOKEN_PROMPT_PAIRS,
    get_duplicate_token_data,
    get_duplicate_token_prompts,
)


def __getattr__(name):
    if name == "generate_induction_data":
        from .induction_data import generate_induction_data

        return generate_induction_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_induction_data",
    "CLEAN_PROMPTS",
    "CORRUPTED_PROMPTS",
    "DUPLICATE_TOKEN_PROMPT_PAIRS",
    "get_duplicate_token_prompts",
    "get_duplicate_token_data",
]
