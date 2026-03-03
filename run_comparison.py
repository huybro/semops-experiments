import os
import re
import csv
import time

import lotus
import pandas as pd
import palimpzest as pz
from lotus.models import LM
from palimpzest.constants import Model
from palimpzest.query.processor.config import QueryProcessorConfig

from data_loader import load_fever_claims, load_oracle_wiki_kb
from retrieval import retrieve_for_claims
from universal_prompts import (
    get_prompt,
    lotus_df2text_row,
    install_prompt_overrides,
    install_pz_prompt_overrides,
)


def nle2str(nle, cols):
    """Replicate LOTUS's nle2str: replace {col} with col.capitalize()."""
    d = {col: col.capitalize() for col in cols}
    return nle.format(**d)


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "qwen/Qwen1.5-0.5B-Chat"
N_CLAIMS = 20
K_RETRIEVAL = 3
MAX_TOKENS = 512
VLLM_API_BASE = "http://localhost:8000/v1"

# Instruction templates — {column} stays as column name reference
FILTER_SUPPORT = (
    "{content}\n"
    "Based on the above evidence, the following claim is supported: {claim}"
)
FILTER_RELEVANCE = (
    "The following evidence is relevant to the claim.\n"
    "Evidence: {content}\nClaim: {claim}"
)
MAP_INSTRUCTION = (
    "Given the claim: {claim}\n"
    "Is this claim true or false based on your knowledge? "
    "Answer with exactly TRUE or FALSE, nothing else."
)
MAP_QUERY_INSTRUCTION = (
    "Given the claim: {claim}\n"
    "Write a short factual search query to find evidence about this claim. "
    "Output only the search query, nothing else."
)

# ============================================================
# Setup — LOTUS (monkey-patch to use get_prompt)
# ============================================================
install_prompt_overrides()
lm = LM(model=f"hosted_vllm/{MODEL_NAME}", api_base=VLLM_API_BASE, max_tokens=MAX_TOKENS, temperature=0)
lotus.settings.configure(lm=lm)

# ============================================================
# litellm interceptor — captures LOTUS + rewrites PZ
# ============================================================
import litellm as _litellm
_original_completion = _litellm.completion

captured = []
rewrite_mode = False
current_filter_instruction = None
current_filter_cols = None


def _interceptor(*args, **kwargs):
    kwargs.setdefault("max_tokens", MAX_TOKENS)
    kwargs.setdefault("temperature", 0)
    messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

    # --- Extract claim + content from messages ---
    claim_val = content_val = None

    # Try PZ JSON format: "claim": "...", "content": "..."
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = msg.get("content", "")
        cm = re.search(r'"claim":\s*"(.*?)"', text, re.DOTALL)
        co = re.search(r'"content":\s*"(.*?)"', text, re.DOTALL)
        if cm and co:
            claim_val, content_val = cm.group(1), co.group(1)
            break
        elif cm:
            claim_val = cm.group(1)
            break

    # Try LOTUS format: [Claim]: «...», [Content]: «...»
    if not claim_val:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            text = msg.get("content", "")
            m = re.search(r'\[Claim\]:\s*«(.*?)»', text, re.DOTALL)
            if m:
                claim_val = m.group(1)
            m2 = re.search(r'\[Content\]:\s*«(.*?)»', text, re.DOTALL)
            if m2:
                content_val = m2.group(1)
            if claim_val:
                break

    # --- PZ mode: rewrite to get_prompt() format ---
    if rewrite_mode:
        rebuilt = False

        if claim_val and content_val and current_filter_instruction:
            cols = current_filter_cols or ["content", "claim"]
            data = lotus_df2text_row({"content": content_val, "claim": claim_val}, cols)
            formatted_instr = nle2str(current_filter_instruction, cols)
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_filter')
            rebuilt = True

        elif claim_val and not content_val:
            is_verdict = any(
                isinstance(m, dict) and "true" in m.get("content", "").lower()
                and "false" in m.get("content", "").lower()
                for m in messages
            )
            data = lotus_df2text_row({"claim": claim_val}, ["claim"])
            instruction = MAP_INSTRUCTION if is_verdict else MAP_QUERY_INSTRUCTION
            formatted_instr = nle2str(instruction, ["claim"])
            kwargs["messages"] = get_prompt(formatted_instr, data, op='sem_map')
            rebuilt = True

        if rebuilt and len(args) > 1:
            args = (args[0],) + (kwargs["messages"],) + args[2:]

    # --- Capture prompt + output ---
    final_msgs = kwargs.get("messages", messages)
    prompt_text = "\n".join(
        m.get("content", "") for m in final_msgs if isinstance(m, dict)
    )
    result = _original_completion(*args, **kwargs)
    output_text = result.choices[0].message.content if result.choices else ""
    captured.append({
        "input": prompt_text,
        "output": output_text,
        "claim": claim_val or "",
        "content": content_val or "",
    })
    return result


_litellm.completion = _interceptor

# ============================================================
# Setup — PZ
# ============================================================
install_pz_prompt_overrides()

PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
PZ_MODEL.api_base = VLLM_API_BASE
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=False,
    allow_mixtures=False,
    allow_critic=False,
    allow_split_merge=False,
    verbose=False,
)

# ============================================================
# Load Data
# ============================================================
claims_df = load_fever_claims(n=N_CLAIMS)
wiki_df = load_oracle_wiki_kb(claims_split="labelled_dev", n_claims=N_CLAIMS)
claims_df["true_label"] = claims_df["label"].apply(lambda l: l == "SUPPORTS")

joined_df = retrieve_for_claims(claims_df, wiki_df, query_col="claim", K=K_RETRIEVAL)

os.makedirs("logs", exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def write_csv(filepath, rows):
    if not rows:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def find_match(cap_list, claim, content=None):
    """Find captured entry matching this claim (+content for filters)."""
    claim_short = claim[:40]
    for entry in cap_list:
        if claim_short in entry.get("claim", ""):
            if content is None or content[:40] in entry.get("content", ""):
                return entry
    for entry in cap_list:
        if claim_short in entry.get("input", ""):
            if content is None or content[:40] in entry.get("input", ""):
                return entry
    return {"input": "", "output": "", "claim": "", "content": ""}


def pz_map_with_fallback(instruction, claims, col_name, pz_desc):
    """Run PZ sem_map, falling back to direct LLM calls if optimizer crashes."""
    try:
        ds = pz.MemoryDataset(
            id=f"cmp-{col_name}",
            vals=claims[["id", "claim", "label", "true_label"]].to_dict("records"),
        )
        ds = ds.sem_map(
            cols=[{"name": col_name, "type": str, "desc": pz_desc}],
            depends_on=["claim"],
        )
        result = ds.run(config=pz_config, max_quality=True)
        return result.to_df()
    except Exception as e:
        outputs = []
        for _, row in claims.iterrows():
            data = lotus_df2text_row({"claim": row["claim"]}, ["claim"])
            instr = nle2str(instruction, ["claim"])
            msgs = get_prompt(instr, data, op='sem_map')
            res = _original_completion(
                model=f"hosted_vllm/{MODEL_NAME}",
                messages=msgs,
                max_tokens=MAX_TOKENS,
                temperature=0,
                api_base=VLLM_API_BASE,
            )
            output = res.choices[0].message.content if res.choices else ""
            outputs.append(output.strip())
            captured.append({
                "input": "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict)),
                "output": output,
                "claim": row["claim"],
                "content": "",
            })
        df = claims.copy()
        df[col_name] = outputs
        return df


# ============================================================
# Pipeline 1: filter
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 1: filter")
print("=" * 60)

rewrite_mode = False
captured.clear()
t0 = time.time()
df_lotus_f = joined_df.copy().sem_filter(FILTER_SUPPORT)
lotus_filter_time = time.time() - t0
lotus_captured = list(captured)
print(f"  LOTUS: {len(df_lotus_f)} passed ({lotus_filter_time:.1f}s)")

rewrite_mode = True
current_filter_instruction = FILTER_SUPPORT
current_filter_cols = ["content", "claim"]
captured.clear()
t0 = time.time()
ds = pz.MemoryDataset(id="cmp-filter", vals=joined_df.to_dict("records"))
ds = ds.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
pz_df = ds.run(config=pz_config).to_df()
pz_filter_time = time.time() - t0
pz_captured_f = list(captured)
rewrite_mode = False
print(f"  PZ:    {len(pz_df)} passed ({pz_filter_time:.1f}s)")

rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lm = find_match(lotus_captured, row["claim"], row["content"])
    pm = find_match(pz_captured_f, row["claim"], row["content"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/comparison_filter.csv", rows)


# ============================================================
# Pipeline 2: filter → filter
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 2: filter → filter")
print("=" * 60)

rewrite_mode = False
captured.clear()
t0 = time.time()
df_f1 = joined_df.copy().sem_filter(FILTER_RELEVANCE)
lotus_f1_captured = list(captured)
captured.clear()
df_f2 = df_f1.sem_filter(FILTER_SUPPORT)
lotus_f2_captured = list(captured)
lotus_ff_time = time.time() - t0
print(f"  LOTUS: {len(df_f1)}→{len(df_f2)} passed ({lotus_ff_time:.1f}s)")

rewrite_mode = True
current_filter_instruction = FILTER_RELEVANCE
current_filter_cols = ["content", "claim"]
captured.clear()
t0 = time.time()
ds2 = pz.MemoryDataset(id="cmp-ff", vals=joined_df.to_dict("records"))
ds2 = ds2.sem_filter("The following evidence is relevant to the claim", depends_on=["content", "claim"])
pz_f1_df = ds2.run(config=pz_config).to_df()
pz_f1_captured = list(captured)

current_filter_instruction = FILTER_SUPPORT
captured.clear()
if len(pz_f1_df) > 0:
    ds2b = pz.MemoryDataset(id="cmp-ff2", vals=pz_f1_df.to_dict("records"))
    ds2b = ds2b.sem_filter(
        "Based on the evidence, the following claim is supported: the claim states that {claim}",
        depends_on=["content", "claim"],
    )
    pz_f2_df = ds2b.run(config=pz_config).to_df()
else:
    pz_f2_df = pd.DataFrame()
pz_ff_time = time.time() - t0
pz_f2_captured = list(captured)
rewrite_mode = False
print(f"  PZ:    {len(pz_f1_df)}→{len(pz_f2_df)} passed ({pz_ff_time:.1f}s)")

rows = []
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    lf1 = find_match(lotus_f1_captured, row["claim"], row["content"])
    pf1 = find_match(pz_f1_captured, row["claim"], row["content"])
    lf2 = find_match(lotus_f2_captured, row["claim"], row["content"])
    pf2 = find_match(pz_f2_captured, row["claim"], row["content"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80], "evidence": row["content"][:80],
        "lotus_f1_input": lf1["input"], "lotus_f1_output": lf1["output"],
        "pz_f1_input": pf1["input"], "pz_f1_output": pf1["output"],
        "lotus_f2_input": lf2["input"], "lotus_f2_output": lf2["output"],
        "pz_f2_input": pf2["input"], "pz_f2_output": pf2["output"],
    })
write_csv("logs/comparison_filter_filter.csv", rows)


# ============================================================
# Pipeline 3: map only
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 3: map only")
print("=" * 60)

rewrite_mode = False
captured.clear()
t0 = time.time()
df_map = claims_df.copy().sem_map(MAP_INSTRUCTION, suffix="verdict")
lotus_map_time = time.time() - t0
lotus_map_captured = list(captured)
print(f"  LOTUS: {len(df_map)} rows ({lotus_map_time:.1f}s)")

rewrite_mode = True
captured.clear()
t0 = time.time()
pz_map_df = pz_map_with_fallback(
    MAP_INSTRUCTION, claims_df, "verdict",
    "TRUE if the claim is factually correct, FALSE otherwise. Answer with exactly TRUE or FALSE.",
)
pz_map_time = time.time() - t0
pz_map_captured = list(captured)
rewrite_mode = False
print(f"  PZ:    {len(pz_map_df)} rows ({pz_map_time:.1f}s)")

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lm = find_match(lotus_map_captured, row["claim"])
    pm = find_match(pz_map_captured, row["claim"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80],
        "lotus_input": lm["input"], "lotus_output": lm["output"],
        "pz_input": pm["input"], "pz_output": pm["output"],
    })
write_csv("logs/comparison_map.csv", rows)


# ============================================================
# Pipeline 4: map → filter
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 4: map → filter")
print("=" * 60)

# LOTUS: map → retrieve → filter
rewrite_mode = False
captured.clear()
t0 = time.time()
df_mf = claims_df.copy().sem_map(MAP_QUERY_INSTRUCTION, suffix="search_query")
lotus_mq_captured = list(captured)
mf_retrieved = retrieve_for_claims(df_mf, wiki_df, query_col="search_query", K=K_RETRIEVAL)
captured.clear()
df_mf_verified = mf_retrieved.sem_filter(FILTER_SUPPORT)
lotus_mf_filter_captured = list(captured)
lotus_mf_time = time.time() - t0
print(f"  LOTUS: map={len(df_mf)}, filter={len(df_mf_verified)} passed ({lotus_mf_time:.1f}s)")

# PZ: map → retrieve → filter
rewrite_mode = True
captured.clear()
t0 = time.time()
pz_mf_map_df = pz_map_with_fallback(
    MAP_QUERY_INSTRUCTION, claims_df, "search_query",
    "A short factual search query to find evidence about the claim",
)
pz_mq_captured = list(captured)

pz_mf_claims = claims_df.copy()
pz_mf_claims["search_query"] = pz_mf_map_df["search_query"].tolist()[:len(pz_mf_claims)]
pz_mf_retrieved = retrieve_for_claims(pz_mf_claims, wiki_df, query_col="search_query", K=K_RETRIEVAL)

current_filter_instruction = FILTER_SUPPORT
current_filter_cols = ["content", "claim"]
captured.clear()
ds4f = pz.MemoryDataset(id="cmp-mf-filter", vals=pz_mf_retrieved.to_dict("records"))
ds4f = ds4f.sem_filter(
    "Based on the evidence, the following claim is supported: the claim states that {claim}",
    depends_on=["content", "claim"],
)
pz_mf_filter_df = ds4f.run(config=pz_config).to_df()
pz_mf_filter_captured = list(captured)
pz_mf_time = time.time() - t0
rewrite_mode = False
print(f"  PZ:    map={len(pz_mf_map_df)}, filter={len(pz_mf_filter_df)} passed ({pz_mf_time:.1f}s)")

rows = []
for i in range(len(claims_df)):
    row = claims_df.iloc[i]
    lm = find_match(lotus_mq_captured, row["claim"])
    pm = find_match(pz_mq_captured, row["claim"])
    rows.append({
        "tuple": i, "claim": row["claim"][:80],
        "lotus_map_input": lm["input"], "lotus_map_output": lm["output"],
        "pz_map_input": pm["input"], "pz_map_output": pm["output"],
    })
write_csv("logs/comparison_map_filter.csv", rows)

