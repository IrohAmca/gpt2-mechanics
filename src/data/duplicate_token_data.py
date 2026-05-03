"""Prompt pairs and tokenization helpers for duplicate-token-head analysis."""

_RAW_DUPLICATE_TOKEN_PROMPT_PAIRS = [
    {
        "clean": "John and Mary went to the store. John gave a drink to",
        "corrupted": "John and Mary went to the store. Paul gave a drink to",
        "first_name": "John",
        "second_name": "Mary",
        "clean_repeated_name": "John",
        "corrupted_name": "Paul",
        "structure": "ABA",
    },
    {
        "clean": "Alice and David walked to the park. Alice handed a book to",
        "corrupted": "Alice and David walked to the park. Sarah handed a book to",
        "first_name": "Alice",
        "second_name": "David",
        "clean_repeated_name": "Alice",
        "corrupted_name": "Sarah",
        "structure": "ABA",
    },
    {
        "clean": "Michael and Laura arrived at the office. Michael sent a letter to",
        "corrupted": "Michael and Laura arrived at the office. Kevin sent a letter to",
        "first_name": "Michael",
        "second_name": "Laura",
        "clean_repeated_name": "Michael",
        "corrupted_name": "Kevin",
        "structure": "ABA",
    },
    {
        "clean": "Emma and Robert waited near the station. Emma passed the ticket to",
        "corrupted": "Emma and Robert waited near the station. Helen passed the ticket to",
        "first_name": "Emma",
        "second_name": "Robert",
        "clean_repeated_name": "Emma",
        "corrupted_name": "Helen",
        "structure": "ABA",
    },
    {
        "clean": "James and Linda sat inside the cafe. James showed the photo to",
        "corrupted": "James and Linda sat inside the cafe. Brian showed the photo to",
        "first_name": "James",
        "second_name": "Linda",
        "clean_repeated_name": "James",
        "corrupted_name": "Brian",
        "structure": "ABA",
    },
    {
        "clean": "Susan and William entered the museum. Susan gave the map to",
        "corrupted": "Susan and William entered the museum. Nancy gave the map to",
        "first_name": "Susan",
        "second_name": "William",
        "clean_repeated_name": "Susan",
        "corrupted_name": "Nancy",
        "structure": "ABA",
    },
    {
        "clean": "George and Patricia stood by the window. George offered a cookie to",
        "corrupted": "George and Patricia stood by the window. Steven offered a cookie to",
        "first_name": "George",
        "second_name": "Patricia",
        "clean_repeated_name": "George",
        "corrupted_name": "Steven",
        "structure": "ABA",
    },
    {
        "clean": "Jennifer and Thomas met at the library. Jennifer returned the pen to",
        "corrupted": "Jennifer and Thomas met at the library. Carol returned the pen to",
        "first_name": "Jennifer",
        "second_name": "Thomas",
        "clean_repeated_name": "Jennifer",
        "corrupted_name": "Carol",
        "structure": "ABA",
    },
    {
        "clean": "Daniel and Barbara crossed the street. Daniel tossed the ball to",
        "corrupted": "Daniel and Barbara crossed the street. Mark tossed the ball to",
        "first_name": "Daniel",
        "second_name": "Barbara",
        "clean_repeated_name": "Daniel",
        "corrupted_name": "Mark",
        "structure": "ABA",
    },
    {
        "clean": "Elizabeth and Richard stayed at the hotel. Elizabeth mailed a postcard to",
        "corrupted": "Elizabeth and Richard stayed at the hotel. Karen mailed a postcard to",
        "first_name": "Elizabeth",
        "second_name": "Richard",
        "clean_repeated_name": "Elizabeth",
        "corrupted_name": "Karen",
        "structure": "ABA",
    },
    {
        "clean": "Joseph and Margaret visited the garden. Joseph brought a flower to",
        "corrupted": "Joseph and Margaret visited the garden. Gary brought a flower to",
        "first_name": "Joseph",
        "second_name": "Margaret",
        "clean_repeated_name": "Joseph",
        "corrupted_name": "Gary",
        "structure": "ABA",
    },
    {
        "clean": "Dorothy and Charles worked in the kitchen. Dorothy poured tea for",
        "corrupted": "Dorothy and Charles worked in the kitchen. Betty poured tea for",
        "first_name": "Dorothy",
        "second_name": "Charles",
        "clean_repeated_name": "Dorothy",
        "corrupted_name": "Betty",
        "structure": "ABA",
    },
    {
        "clean": "Christopher and Sandra waited in the lobby. Christopher saved a seat for",
        "corrupted": "Christopher and Sandra waited in the lobby. Edward saved a seat for",
        "first_name": "Christopher",
        "second_name": "Sandra",
        "clean_repeated_name": "Christopher",
        "corrupted_name": "Edward",
        "structure": "ABA",
    },
    {
        "clean": "Ashley and Matthew walked beside the river. Ashley carried the bag for",
        "corrupted": "Ashley and Matthew walked beside the river. Donna carried the bag for",
        "first_name": "Ashley",
        "second_name": "Matthew",
        "clean_repeated_name": "Ashley",
        "corrupted_name": "Donna",
        "structure": "ABA",
    },
    {
        "clean": "Joshua and Kimberly reached the airport. Joshua bought a sandwich for",
        "corrupted": "Joshua and Kimberly reached the airport. Ronald bought a sandwich for",
        "first_name": "Joshua",
        "second_name": "Kimberly",
        "clean_repeated_name": "Joshua",
        "corrupted_name": "Ronald",
        "structure": "ABA",
    },
    {
        "clean": "Emily and Andrew stopped near the theater. Emily found a scarf for",
        "corrupted": "Emily and Andrew stopped near the theater. Lisa found a scarf for",
        "first_name": "Emily",
        "second_name": "Andrew",
        "clean_repeated_name": "Emily",
        "corrupted_name": "Lisa",
        "structure": "ABA",
    },
    {
        "clean": "Anthony and Michelle moved toward the classroom. Anthony opened the door for",
        "corrupted": "Anthony and Michelle moved toward the classroom. Jason opened the door for",
        "first_name": "Anthony",
        "second_name": "Michelle",
        "clean_repeated_name": "Anthony",
        "corrupted_name": "Jason",
        "structure": "ABA",
    },
    {
        "clean": "Amanda and Donald stood outside the bakery. Amanda paid the clerk for",
        "corrupted": "Amanda and Donald stood outside the bakery. Shirley paid the clerk for",
        "first_name": "Amanda",
        "second_name": "Donald",
        "clean_repeated_name": "Amanda",
        "corrupted_name": "Shirley",
        "structure": "ABA",
    },
    {
        "clean": "Ryan and Melissa looked around the gallery. Ryan described the painting to",
        "corrupted": "Ryan and Melissa looked around the gallery. Frank described the painting to",
        "first_name": "Ryan",
        "second_name": "Melissa",
        "clean_repeated_name": "Ryan",
        "corrupted_name": "Frank",
        "structure": "ABA",
    },
    {
        "clean": "Deborah and Justin hurried into the train. Deborah saved the ticket for",
        "corrupted": "Deborah and Justin hurried into the train. Ruth saved the ticket for",
        "first_name": "Deborah",
        "second_name": "Justin",
        "clean_repeated_name": "Deborah",
        "corrupted_name": "Ruth",
        "structure": "ABA",
    },
    {
        "clean": "Brandon and Stephanie gathered in the hallway. Brandon wrote a note to",
        "corrupted": "Brandon and Stephanie gathered in the hallway. Larry wrote a note to",
        "first_name": "Brandon",
        "second_name": "Stephanie",
        "clean_repeated_name": "Brandon",
        "corrupted_name": "Larry",
        "structure": "ABA",
    },
    {
        "clean": "Rebecca and Eric worked behind the counter. Rebecca sold a card to",
        "corrupted": "Rebecca and Eric worked behind the counter. Angela sold a card to",
        "first_name": "Rebecca",
        "second_name": "Eric",
        "clean_repeated_name": "Rebecca",
        "corrupted_name": "Angela",
        "structure": "ABA",
    },
    {
        "clean": "Timothy and Sharon stopped at the market. Timothy handed the receipt to",
        "corrupted": "Timothy and Sharon stopped at the market. Scott handed the receipt to",
        "first_name": "Timothy",
        "second_name": "Sharon",
        "clean_repeated_name": "Timothy",
        "corrupted_name": "Scott",
        "structure": "ABA",
    },
    {
        "clean": "Cynthia and Kenneth relaxed near the fountain. Cynthia showed the coin to",
        "corrupted": "Cynthia and Kenneth relaxed near the fountain. Maria showed the coin to",
        "first_name": "Cynthia",
        "second_name": "Kenneth",
        "clean_repeated_name": "Cynthia",
        "corrupted_name": "Maria",
        "structure": "ABA",
    },
    {
        "clean": "Nicholas and Kathleen entered the studio. Nicholas played a song for",
        "corrupted": "Nicholas and Kathleen entered the studio. Jeffrey played a song for",
        "first_name": "Nicholas",
        "second_name": "Kathleen",
        "clean_repeated_name": "Nicholas",
        "corrupted_name": "Jeffrey",
        "structure": "ABA",
    },
    {
        "clean": "Amy and Stephen parked near the stadium. Amy bought a flag for",
        "corrupted": "Amy and Stephen parked near the stadium. Christine bought a flag for",
        "first_name": "Amy",
        "second_name": "Stephen",
        "clean_repeated_name": "Amy",
        "corrupted_name": "Christine",
        "structure": "ABA",
    },
    {
        "clean": "Jacob and Anna waited outside the school. Jacob packed a lunch for",
        "corrupted": "Jacob and Anna waited outside the school. Peter packed a lunch for",
        "first_name": "Jacob",
        "second_name": "Anna",
        "clean_repeated_name": "Jacob",
        "corrupted_name": "Peter",
        "structure": "ABA",
    },
    {
        "clean": "Rachel and Jonathan worked near the bridge. Rachel carried a lantern for",
        "corrupted": "Rachel and Jonathan worked near the bridge. Diane carried a lantern for",
        "first_name": "Rachel",
        "second_name": "Jonathan",
        "clean_repeated_name": "Rachel",
        "corrupted_name": "Diane",
        "structure": "ABA",
    },
    {
        "clean": "Tyler and Katherine sat beside the fireplace. Tyler poured a glass for",
        "corrupted": "Tyler and Katherine sat beside the fireplace. Samuel poured a glass for",
        "first_name": "Tyler",
        "second_name": "Katherine",
        "clean_repeated_name": "Tyler",
        "corrupted_name": "Samuel",
        "structure": "ABA",
    },
    {
        "clean": "Nicole and Benjamin waited at the clinic. Nicole opened a magazine for",
        "corrupted": "Nicole and Benjamin waited at the clinic. Julie opened a magazine for",
        "first_name": "Nicole",
        "second_name": "Benjamin",
        "clean_repeated_name": "Nicole",
        "corrupted_name": "Julie",
        "structure": "ABA",
    },
]

for pair in _RAW_DUPLICATE_TOKEN_PROMPT_PAIRS:
    first_name = pair["first_name"]
    second_name = pair["second_name"]
    corrupted_name = pair["corrupted_name"]
    pair["clean"] = (
        f"Yesterday, {first_name} and {second_name} went to the store. "
        f"{first_name} gave a drink to"
    )
    pair["corrupted"] = (
        f"Yesterday, {first_name} and {second_name} went to the store. "
        f"{corrupted_name} gave a drink to"
    )
    pair["clean_repeated_name"] = first_name
    pair["correct_name"] = second_name
    pair["incorrect_name"] = first_name
    pair["structure"] = "ABA"

DUPLICATE_TOKEN_PROMPT_PAIRS = _RAW_DUPLICATE_TOKEN_PROMPT_PAIRS

CLEAN_PROMPTS = [pair["clean"] for pair in DUPLICATE_TOKEN_PROMPT_PAIRS]
CORRUPTED_PROMPTS = [pair["corrupted"] for pair in DUPLICATE_TOKEN_PROMPT_PAIRS]


def _encode(tokenizer, text):
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _find_token_span(token_ids, pattern_ids):
    pattern_len = len(pattern_ids)
    for start_idx in range(len(token_ids) - pattern_len + 1):
        if token_ids[start_idx : start_idx + pattern_len] == pattern_ids:
            return list(range(start_idx, start_idx + pattern_len))
    raise ValueError(f"Could not find token pattern {pattern_ids} in tokenized prompt.")


def _single_token_index(token_span):
    return token_span[0] if len(token_span) == 1 else None


def _single_token_id(tokenizer, text):
    token_ids = _encode(tokenizer, " " + text)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to be one token, got {token_ids}.")
    return token_ids[0]


def _token_span_from_offsets(tokenizer, text, char_start, char_end):
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except TypeError:
        return None

    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    if offsets is None:
        return None

    span = [
        token_idx
        for token_idx, (offset_start, offset_end) in enumerate(offsets)
        if offset_start < char_end and offset_end > char_start
    ]
    return span or None


def _token_span_from_prefix(tokenizer, text, char_start, char_end):
    token_ids = _encode(tokenizer, text[:char_end])
    prefix_ids = _encode(tokenizer, text[:char_start])
    span = list(range(len(prefix_ids), len(token_ids)))
    if span:
        return span

    name_with_prefix_space = text[char_start - 1 : char_end]
    return _find_token_span(token_ids, _encode(tokenizer, name_with_prefix_space))


def _token_span_for_char_span(tokenizer, text, char_start, char_end):
    return _token_span_from_offsets(
        tokenizer,
        text,
        char_start,
        char_end,
    ) or _token_span_from_prefix(tokenizer, text, char_start, char_end)


def _with_device(batch, device):
    if device is None or not hasattr(batch, "to"):
        return batch
    return batch.to(device)


def _prepare_padding(tokenizer, padding):
    if not padding:
        return
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        raise ValueError("Tokenizer needs a pad_token for padded duplicate-token data.")
    tokenizer.pad_token = eos_token


def _build_token_metadata(tokenizer, prompt_pairs):
    metadata = []
    for pair in prompt_pairs:
        first_name = pair["first_name"]
        corrupted_name = pair["corrupted_name"]
        first_marker = f"{first_name} and"
        second_marker = f". {first_name} "
        clean = pair["clean"]
        corrupted = pair["corrupted"]

        first_char_start = clean.index(first_marker)
        second_char_start = clean.index(second_marker) + 2
        first_char_end = first_char_start + len(first_name)
        second_char_end = second_char_start + len(first_name)
        corrupted_subject_char_start = corrupted.index(f". {corrupted_name} ") + 2
        corrupted_subject_char_end = corrupted_subject_char_start + len(corrupted_name)

        first_token_span = _token_span_for_char_span(
            tokenizer,
            clean,
            first_char_start,
            first_char_end,
        )
        second_token_span = _token_span_for_char_span(
            tokenizer,
            clean,
            second_char_start,
            second_char_end,
        )
        corrupted_subject_token_span = _token_span_for_char_span(
            tokenizer,
            corrupted,
            corrupted_subject_char_start,
            corrupted_subject_char_end,
        )
        clean_token_ids = _encode(tokenizer, clean)
        corrupted_token_ids = _encode(tokenizer, corrupted)

        metadata.append(
            {
                **pair,
                "prediction_token_index": len(clean_token_ids) - 1,
                "first_repeat_char_start": first_char_start,
                "first_repeat_char_end": first_char_end,
                "second_repeat_char_start": second_char_start,
                "second_repeat_char_end": second_char_end,
                "corrupted_subject_char_start": corrupted_subject_char_start,
                "corrupted_subject_char_end": corrupted_subject_char_end,
                "first_repeat_token_span": first_token_span,
                "second_repeat_token_span": second_token_span,
                "corrupted_subject_token_span": corrupted_subject_token_span,
                "first_repeat_token_index": _single_token_index(first_token_span),
                "second_repeat_token_index": _single_token_index(second_token_span),
                "corrupted_subject_token_index": _single_token_index(
                    corrupted_subject_token_span
                ),
                "first_repeat_token_ids": [
                    clean_token_ids[token_idx] for token_idx in first_token_span
                ],
                "second_repeat_token_ids": [
                    clean_token_ids[token_idx] for token_idx in second_token_span
                ],
                "corrupted_subject_token_ids": [
                    corrupted_token_ids[token_idx]
                    for token_idx in corrupted_subject_token_span
                ],
                "correct_token_id": _single_token_id(tokenizer, pair["correct_name"]),
                "incorrect_token_id": _single_token_id(
                    tokenizer, pair["incorrect_name"]
                ),
                "corrupted_subject_token_id": _single_token_id(
                    tokenizer, pair["corrupted_name"]
                ),
            }
        )
    return metadata


def get_duplicate_token_prompts():
    """Return clean/corrupted prompt lists and per-example metadata."""
    return {
        "clean_prompts": CLEAN_PROMPTS,
        "corrupted_prompts": CORRUPTED_PROMPTS,
        "metadata": DUPLICATE_TOKEN_PROMPT_PAIRS,
    }


def get_duplicate_token_data(
    tokenizer,
    return_tensors="pt",
    padding=True,
    truncation=False,
    device=None,
):
    """Tokenize clean/corrupted prompt pairs and return repeat-token metadata."""
    _prepare_padding(tokenizer, padding)
    clean_tokens = tokenizer(
        CLEAN_PROMPTS,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
    )
    corrupted_tokens = tokenizer(
        CORRUPTED_PROMPTS,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
    )

    return {
        "clean_tokens": _with_device(clean_tokens, device),
        "corrupted_tokens": _with_device(corrupted_tokens, device),
        "metadata": _build_token_metadata(tokenizer, DUPLICATE_TOKEN_PROMPT_PAIRS),
    }
