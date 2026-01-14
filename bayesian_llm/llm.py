from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LoadedHFModel:
    model: "object"
    tokenizer: "object"


def load_hf_causal_lm(
    model_id: str,
    *,
    torch_dtype: "object" = None,
    device_map: str | dict | None = "auto",
):
    """
    Loads a HuggingFace causal LM + tokenizer.

    Notes:
      - For Meta Llama 3.x models on HF, you may need to set HF_TOKEN and have accepted the license.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return LoadedHFModel(model=model, tokenizer=tokenizer)


def next_token_distribution(
    model,
    tokenizer,
    prompt: str,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1).detach().float().cpu().numpy()
    return probs


def _token_ids_for_variants(tokenizer, variants: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        # We only support single-token variants for next-token probabilities.
        if len(enc) != 1:
            continue
        ids.append(int(enc[0]))
    return sorted(set(ids))


def normalized_next_token_prob(
    model,
    tokenizer,
    prompt: str,
    *,
    a_variants: Iterable[str],
    b_variants: Iterable[str],
    temperature: float = 1.0,
) -> float:
    """
    Returns P(A | A or B) using the model's next-token distribution.

    Variants exist because Llama-style tokenizers often distinguish "X" vs " X".
    """
    probs = next_token_distribution(model, tokenizer, prompt, temperature=temperature)

    a_ids = _token_ids_for_variants(tokenizer, a_variants)
    b_ids = _token_ids_for_variants(tokenizer, b_variants)
    if len(a_ids) == 0 or len(b_ids) == 0:
        raise ValueError("Could not tokenize one of the candidate variants.")

    p_a = float(probs[a_ids].sum())
    p_b = float(probs[b_ids].sum())
    denom = p_a + p_b
    return p_a / denom if denom > 0 else 0.5
