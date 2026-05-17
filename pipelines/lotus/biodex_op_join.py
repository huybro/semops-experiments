
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import pandas as pd

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.types import CascadeArgs, ProxyModel
from lotus.vector_store import FaissVS
from pipelines.cli_utils import parse_vllm_args


total_start = time.perf_counter()
MAX_TOKENS = 8192
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3

DATA_ROOT = Path(PROJECT_ROOT) / "data"
ARTICLE_DIR = Path(os.environ.get("BIODEX_ARTICLE_DIR", DATA_ROOT / "articles"))
REACTION_DIR = Path(os.environ.get("BIODEX_REACTION_DIR", DATA_ROOT / "reactions"))
ARTICLE_LIMIT = int(os.environ.get("BIODEX_ARTICLE_LIMIT", "250"))
REACTION_LIMIT = int(os.environ.get("BIODEX_REACTION_LIMIT", "11271"))
MAX_ARTICLE_CHARS = int(os.environ.get("BIODEX_MAX_ARTICLE_CHARS", "12000"))
EMBEDDING_MODEL = os.environ.get("LOTUS_RM_MODEL", "intfloat/e5-base-v2")
CASCADE_RECALL_TARGET = float(os.environ.get("BIODEX_CASCADE_RECALL_TARGET", "0.8"))
CASCADE_PRECISION_TARGET = float(os.environ.get("BIODEX_CASCADE_PRECISION_TARGET", "0.8"))
CASCADE_SAMPLING_PERCENTAGE = float(os.environ.get("BIODEX_CASCADE_SAMPLING_PERCENTAGE", "0.0001"))
CASCADE_STRATEGY = os.environ.get("BIODEX_JOIN_CASCADE_STRATEGY") or None


def optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None

CASCADE_POS_THRESHOLD = 1
CASCADE_NEG_THRESHOLD = 0.825 #optional_float_env("BIODEX_JOIN_CASCADE_NEG_THRESHOLD")

MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
rm = SentenceTransformersRM(model=EMBEDDING_MODEL)
vs = FaissVS()

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


def numbered_text_files(folder: Path, limit: int) -> list[Path]:
    files = sorted(
        folder.glob("*.txt"),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )
    return files[:limit] if limit > 0 else files


def load_articles(folder: Path, limit: int) -> pd.DataFrame:
    rows = []
    for path in numbered_text_files(folder, limit):
        rows.append(
            {
                "article_id": path.stem,
                "article": path.read_text(encoding="utf-8").strip()[:MAX_ARTICLE_CHARS],
            }
        )
    return pd.DataFrame(rows)


def load_reactions(folder: Path, limit: int) -> pd.DataFrame:
    rows = []
    for path in numbered_text_files(folder, limit):
        label = " ".join(path.read_text(encoding="utf-8").strip().split())
        if label:
            rows.append({"reaction_id": path.stem, "reaction_label": label})
    return pd.DataFrame(rows)


df_articles = load_articles(ARTICLE_DIR, ARTICLE_LIMIT)
df_reactions = load_reactions(REACTION_DIR, REACTION_LIMIT)
print(f"loaded {len(df_articles)} articles from {ARTICLE_DIR}")
print(f"loaded {len(df_reactions)} reactions from {REACTION_DIR}")
print(f"load_time_sec: {time.perf_counter() - total_start:.3f}")
print(f"cascade_recall_target: {CASCADE_RECALL_TARGET}")
print(f"cascade_precision_target: {CASCADE_PRECISION_TARGET}")
print(f"cascade_sampling_percentage: {CASCADE_SAMPLING_PERCENTAGE}")

join_instruction = (
    "{article:left}{reaction_label:right}"
    "Does the biomedical article describe a patient who experienced this adverse drug reaction?"
)

cascade_kwargs = {
    "recall_target": CASCADE_RECALL_TARGET,
    "precision_target": CASCADE_PRECISION_TARGET,
    "sampling_percentage": CASCADE_SAMPLING_PERCENTAGE,
    "failure_probability": 0.2,
    "min_join_cascade_size": 1,
    "map_instruction": (
        "{article}"
        "Extract adverse drug reaction terms that are explicitly described for the patient in this article. "
        "Always write your answer as a list of 2-10 comma-separated reaction labels."
    ),
    "proxy_model": ProxyModel.EMBEDDING_MODEL,
}

if CASCADE_STRATEGY is not None:
    cascade_kwargs["join_cascade_strategy"] = CASCADE_STRATEGY
if CASCADE_POS_THRESHOLD is not None or CASCADE_NEG_THRESHOLD is not None:
    cascade_kwargs["join_cascade_pos_threshold"] = CASCADE_POS_THRESHOLD
    cascade_kwargs["join_cascade_neg_threshold"] = CASCADE_NEG_THRESHOLD

cascade_args = CascadeArgs(
    **cascade_kwargs,
)

join_start = time.perf_counter()
res, stats = df_articles.sem_join(
    df_reactions,
    join_instruction,
    cascade_args=cascade_args,
    return_stats=True,
)
join_elapsed = time.perf_counter() - join_start
total_elapsed = time.perf_counter() - total_start

print("res")
print(res[["article_id", "reaction_id", "reaction_label"]])
print("stats")
print(stats)
print(f"join_time_sec: {join_elapsed:.3f}")
print(f"total_time_sec: {total_elapsed:.3f}")
