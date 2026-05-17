import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from pipelines import llm_intercepter, scenarios
from pipelines.cli_utils import parse_vllm_args
from data_utils import write_csv


project = "lotus"
FILTER_MAX_TOKENS = 8
MAX_TOKENS = 4096
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3

BIODEX_SPLIT = os.environ.get("BIODEX_SPLIT", "test")
BIODEX_ARTICLE_LIMIT = int(os.environ.get("BIODEX_ARTICLE_LIMIT", "250"))
BIODEX_LABEL_LIMIT = int(os.environ.get("BIODEX_LABEL_LIMIT", "0"))
BIODEX_TOP_K = int(os.environ.get("BIODEX_TOP_K", "20"))
BIODEX_USE_FULLTEXT = os.environ.get("BIODEX_USE_FULLTEXT", "1") == "1"
BIODEX_MAX_ARTICLE_CHARS = int(os.environ.get("BIODEX_MAX_ARTICLE_CHARS", "12000"))
EMBEDDING_MODEL = os.environ.get("LOTUS_RM_MODEL", "intfloat/e5-base-v2")
LABEL_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "biodex_reaction_labels_faiss_index",
)

MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def normalize_reaction(label):
    return " ".join(str(label).strip().split())


def reaction_labels(example):
    reactions = example.get("reactions") or ""
    labels = [normalize_reaction(label) for label in reactions.split(",") if normalize_reaction(label)]
    if labels:
        return labels

    unmerged = example.get("reactions_unmerged") or []
    labels = []
    for label_group in unmerged:
        labels.extend(
            normalize_reaction(label)
            for label in str(label_group).split(",")
            if normalize_reaction(label)
        )
    return labels


def article_text(example):
    parts = []
    title = (example.get("title") or "").strip()
    abstract = (example.get("abstract") or "").strip()
    fulltext = (example.get("fulltext_processed") or example.get("fulltext") or "").strip()

    if BIODEX_USE_FULLTEXT and fulltext:
        return fulltext[:BIODEX_MAX_ARTICLE_CHARS]

    if title:
        parts.append(f"TITLE: {title}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")

    text = "\n\n".join(parts)
    return text[:BIODEX_MAX_ARTICLE_CHARS]


def load_biodex_articles(split, limit):
    ds = load_dataset("BioDEX/BioDEX-Reactions", split=f"{split}[:{limit}]")
    rows = []
    for idx, example in enumerate(ds):
        labels = reaction_labels(example)
        rows.append(
            {
                "article_id": idx,
                "pmid": example.get("pmid"),
                "article": article_text(example),
                "gold_reactions": "; ".join(sorted(set(labels))),
            }
        )
    return pd.DataFrame(rows)


def load_biodex_label_space(label_limit):
    labels = set()
    for split in ("train", "validation", "test"):
        ds = load_dataset("BioDEX/BioDEX-Reactions", split=split)
        for example in ds:
            labels.update(reaction_labels(example))

    label_rows = [{"label_id": i, "reaction_label": label} for i, label in enumerate(sorted(labels))]
    if label_limit > 0:
        label_rows = label_rows[:label_limit]
    return pd.DataFrame(label_rows)


rm = SentenceTransformersRM(model=EMBEDDING_MODEL)
vs = FaissVS()

_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=FILTER_MAX_TOKENS,
    temperature=0,
    top_p=1,
    seed=42,
    frequency_penalty=FREQUENCY_PENALTY,
    repetition_penalty=REPETITION_PENALTY,
)
lotus.settings.configure(lm=_lotus_lm, rm=rm, vs=vs)

t0 = time.time()
df_articles = load_biodex_articles(BIODEX_SPLIT, BIODEX_ARTICLE_LIMIT)
df_labels = load_biodex_label_space(BIODEX_LABEL_LIMIT)
print(
    f"  LOTUS: loaded {len(df_articles)} BioDEX articles and {len(df_labels)} reaction labels "
    f"({time.time() - t0:.1f}s)"
)

df_labels = df_labels.sem_index("reaction_label", LABEL_INDEX_DIR)
print(f"  LOTUS: built reaction label index at {LABEL_INDEX_DIR} ({time.time() - t0:.1f}s)")

log = []
params = {
    "log": log,
    "max_tokens": FILTER_MAX_TOKENS,
    "tokenizer": tokenizer,
    "seed": 42,
    "frequency_penalty": FREQUENCY_PENALTY,
    "repetition_penalty": REPETITION_PENALTY,
}
llm_intercepter.set_intercept(**params)

df_candidates = df_articles.sem_sim_join(
    df_labels,
    left_on="article",
    right_on="reaction_label",
    K=BIODEX_TOP_K,
    rsuffix="_reaction",
).reset_index(drop=True)
print(
    f"  LOTUS: sim_join produced {len(df_candidates)} candidate article/reaction pairs "
    f"({time.time() - t0:.1f}s)"
)

df_joined = df_candidates.sem_filter(scenarios.BIODEX_CASE_1_JOIN)
print(
    f"  LOTUS: LLM join kept {len(df_joined)}/{len(df_candidates)} candidate pairs "
    f"({time.time() - t0:.1f}s)"
)

_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
    top_p=1,
    seed=42,
    frequency_penalty=FREQUENCY_PENALTY,
    repetition_penalty=REPETITION_PENALTY,
)
lotus.settings.configure(lm=_lotus_lm, rm=rm, vs=vs)
params = {
    "log": log,
    "max_tokens": MAX_TOKENS,
    "tokenizer": tokenizer,
    "seed": 42,
    "frequency_penalty": FREQUENCY_PENALTY,
    "repetition_penalty": REPETITION_PENALTY,
}
llm_intercepter.set_intercept(**params)

df_mapped = df_joined.sem_map(scenarios.BIODEX_CASE_1_MAP)
print(f"  LOTUS: map {len(df_mapped)}/{len(df_joined)} kept pairs ({time.time() - t0:.1f}s)")
print(df_mapped[["pmid", "reaction_label", "gold_reactions"]].head(20))

rows = []
join_log_len = len(df_candidates)
for i in range(min(join_log_len, len(log))):
    rows.append(
        {
            "op": "join_filter",
            "lotus_input": log[i]["input"],
            "lotus_output": log[i]["output"],
        }
    )

for i in range(join_log_len, len(log)):
    rows.append(
        {
            "op": "map",
            "lotus_input": log[i]["input"],
            "lotus_output": log[i]["output"],
        }
    )

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
