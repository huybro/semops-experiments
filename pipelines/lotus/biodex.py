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
from lotus.models import LM
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
BIODEX_LABEL_SOURCE = os.environ.get("BIODEX_LABEL_SOURCE", "raw")
BIODEX_USE_FULLTEXT = os.environ.get("BIODEX_USE_FULLTEXT", "1") == "1"
BIODEX_MAX_ARTICLE_CHARS = int(os.environ.get("BIODEX_MAX_ARTICLE_CHARS", "12000"))

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
    title = (example.get("title") or "").strip()
    abstract = (example.get("abstract") or "").strip()
    fulltext = (example.get("fulltext_processed") or example.get("fulltext") or "").strip()

    if BIODEX_USE_FULLTEXT and fulltext:
        return fulltext[:BIODEX_MAX_ARTICLE_CHARS]

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    return "\n\n".join(parts)[:BIODEX_MAX_ARTICLE_CHARS]


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


def load_reactions_dataset_labels():
    labels = set()
    for split in ("train", "validation", "test"):
        ds = load_dataset("BioDEX/BioDEX-Reactions", split=split)
        for example in ds:
            labels.update(reaction_labels(example))
    return pd.DataFrame(
        {"label_id": idx, "reaction_label": label}
        for idx, label in enumerate(sorted(labels))
    )


def raw_report_reaction_labels(report):
    patient = report.get("patient") or {}
    reactions = patient.get("reaction") or []
    if isinstance(reactions, dict):
        reactions = [reactions]

    labels = []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        label = normalize_reaction(reaction.get("reactionmeddrapt"))
        if label and label.lower() != "none":
            labels.append(label)
    return labels


def load_raw_biodex_reaction_labels():
    labels = set()
    ds = load_dataset("BioDEX/raw_dataset", split="train")
    for example in ds:
        reports = example.get("reports") or []
        for report in reports:
            if isinstance(report, dict):
                labels.update(raw_report_reaction_labels(report))
    return pd.DataFrame(
        {"label_id": idx, "reaction_label": label}
        for idx, label in enumerate(sorted(labels))
    )


def load_all_biodex_reaction_labels(label_source):
    if label_source == "raw":
        return load_raw_biodex_reaction_labels()
    if label_source == "reactions":
        return load_reactions_dataset_labels()
    raise ValueError("BIODEX_LABEL_SOURCE must be 'raw' or 'reactions'")


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
lotus.settings.configure(lm=_lotus_lm)

t0 = time.time()
df_articles = load_biodex_articles(BIODEX_SPLIT, BIODEX_ARTICLE_LIMIT)
df_labels = load_all_biodex_reaction_labels(BIODEX_LABEL_SOURCE)
print(
    f"  LOTUS: loaded {len(df_articles)} BioDEX articles and {len(df_labels)} reaction labels "
    f"from {BIODEX_LABEL_SOURCE} label source "
    f"({time.time() - t0:.1f}s)"
)
print(f"  LOTUS: exhaustive join will check {len(df_articles) * len(df_labels)} article/reaction pairs")
