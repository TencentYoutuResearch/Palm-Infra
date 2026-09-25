"""Rebuild the Hy4 tokenizer assets carried by a GGUF checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from gguf_reader import GgufFile

_TEMPLATE = Path(__file__).with_name("hy4_tokenizer_template.json")
_NON_SPECIAL_CONTROL = {
    "<｜hy_place▁holder▁no▁13｜>",
    "<｜hy_place▁holder▁no▁14｜>",
}


def tokenizer_from_gguf(gguf: GgufFile) -> tuple[dict, str]:
    metadata = gguf.metadata
    if metadata.get("tokenizer.ggml.model") != "gpt2" or \
            metadata.get("tokenizer.ggml.pre") != "hyv4":
        raise ValueError("GGUF tokenizer is not the supported Hy4 GPT-2 BPE")
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    merges = metadata["tokenizer.ggml.merges"]
    chat_template = metadata["tokenizer.chat_template"]
    if len(tokens) != len(types) or len(set(tokens)) != len(tokens):
        raise ValueError("GGUF tokenizer tokens or token types are inconsistent")
    first_added = next((i for i, typ in enumerate(types) if int(typ) != 1),
                       len(tokens))
    if any(int(typ) != 1 for typ in types[:first_added]) or \
            any(int(typ) not in (3, 4) for typ in types[first_added:]):
        raise ValueError("unsupported GGUF tokenizer token type sequence")
    if first_added != 120000 or len(tokens) != 120832:
        raise ValueError("unexpected Hy4 tokenizer vocabulary size")

    template = json.loads(_TEMPLATE.read_text())
    template["model"]["vocab"] = {token: i for i, token in enumerate(
        tokens[:first_added])}
    pairs = [merge.split(" ", 1) for merge in merges]
    if any(len(pair) != 2 or not all(pair) for pair in pairs):
        raise ValueError("GGUF tokenizer contains a malformed BPE merge")
    template["model"]["merges"] = pairs
    template["added_tokens"] = [
        {
            "id": i,
            "content": tokens[i],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": int(types[i]) == 3 and
                       tokens[i] not in _NON_SPECIAL_CONTROL,
        }
        for i in range(first_added, len(tokens))
    ]
    return template, str(chat_template)
