"""
Shared setup for FEVER experiment pipelines.

Provides:
- Configuration constants and instruction templates
- LOTUS and Palimpzest initialization
- litellm interceptor for prompt capture/rewriting
- Helper functions (write_csv, find_match, pz_map_with_fallback)
- Data loading from pre-retrieved CSV

Usage:
    from experiment_utils import (
        state, joined_df, pz_config, pz,
        FILTER_RELEVANCE, FILTER_SUPPORT, MAP_VERDICT,
        write_csv, find_match, pz_map_with_fallback,
    )
"""
import os
import csv
import lotus
import pandas as pd
import litellm as _litellm
from lotus.models import LM

from transformers import AutoTokenizer
from data_utils import write_csv, load_fever


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_TOKENS = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
VLLM_API_BASE = "http://localhost:8003/v1"

# ============================================================
# Instruction templates
# ============================================================

# Filter: is the evidence relevant to the claim?
FILTER_RELEVANCE = (
    "The sentence can determine whether the claim is true or false.\n"
    "Answer TRUE if the context is sufficient to judge the claim, and FALSE otherwise.\n"
    "Output TRUE or FALSE only."
)

# Filter: does the evidence support the claim?
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)

# Map: based on both claim and evidence, generate a verdict (Claim first, Evidence second)
MAP_VERDICT = (
    "{claim} {content}\n"
    "Explain how the claim can be supported by the evidence.\n"
    "Provide a short explanation in natural language."
)

# Filter: verify verdict against evidence and claim (used after sem_map)
FILTER_VERDICT = (
    "Based on the evidence and verdict, the claim is correctly assessed.\n"
    "Evidence: {content}\nClaim: {claim}\nVerdict: {verdict}"
)

# Enron filter -> filter -> map templates
FILTER_ENRON_FRAUD = (
    "{contents}\n"
    'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy"). '
    "Answer TRUE if it does, FALSE otherwise. Output TRUE or FALSE only.\n"
)

FILTER_ENRON_NOT_NEWS = (
    "{contents}\n"
    "The email is not quoting from a news article or an article written by someone outside of Enron. "
    "Answer TRUE if it is NOT quoting such an article, FALSE otherwise. Output TRUE or FALSE only.\n"
)

MAP_ENRON_EXPLANATION = (
    "{contents}\n"
    "Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context."
)


# ============================================================
# Interceptor state (mutable, shared across pipeline steps)
# ============================================================
logger = []


# FEVER joined_df reused across LOTUS pipelines
FEVER_PATH = os.path.join("data", "fever_claims_with_evidence.csv")
joined_df = load_fever(FEVER_PATH)


# ============================================================
# Setup — LOTUS
# ============================================================
_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
)
lotus.settings.configure(lm=_lotus_lm)



_original_completion = _litellm.completion
 

def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("seed", 42)

    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""

    logger.append({"input": prompt_text, "output": output_text})
    return result


_litellm.completion = _interceptor


