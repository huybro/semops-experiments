import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import pandas as pd
import lotus
from lotus.models import LM
from transformers import AutoTokenizer

from data_utils import write_csv
from pipelines import llm_intercepter, scenarios
from pipelines.cli_utils import parse_vllm_args


project = "lotus"
FILTER_MAX_TOKENS = 8
MAX_TOKENS = 8192
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3
LITELLM_TIMEOUT = 3600

MEDEC_CSV = os.environ.get(
    "MEDEC_CSV",
    os.path.join(PROJECT_ROOT, "data", "MEDEC-Full-TrainingSet-agreement-balanced-1000-with-ErrorType.csv"),
)
MEDEC_LIMIT = int(os.environ.get("MEDEC_LIMIT", "0"))
MEDEC_FILTER_ONLY = os.environ.get("MEDEC_FILTER_ONLY", "0") == "1"

MODEL_NAME = None
VLLM_API_BASE = None
tokenizer = None


def configure_lotus(max_tokens, log):
    global MODEL_NAME, VLLM_API_BASE, tokenizer
    if MODEL_NAME is None or VLLM_API_BASE is None:
        MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lotus_lm = LM(
        model=f"hosted_vllm/{MODEL_NAME}",
        api_base=VLLM_API_BASE,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        seed=42,
        frequency_penalty=FREQUENCY_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        timeout=LITELLM_TIMEOUT,
    )
    lotus.settings.configure(lm=lotus_lm)
    llm_intercepter.set_intercept(
        log=log,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        seed=42,
        timeout=LITELLM_TIMEOUT,
        frequency_penalty=FREQUENCY_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
    )


def load_medec(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if MEDEC_LIMIT > 0:
        df = df.head(MEDEC_LIMIT)
    df = df.reset_index(drop=True)
    df["data"] = df.apply(
        lambda row: (
            f"Text ID: {str(row.get('Text ID', '')).strip()}\n\n"
            f"Clinical note:\n{str(row['Text']).strip()}\n\n"
            f"Numbered sentences:\n{str(row['Sentences']).strip()}"
        ),
        axis=1,
    )
    return df


def print_eval(df_all, df_flagged, df_id, df_correction):
    if "Error Flag" not in df_all.columns:
        return

    predicted_error_ids = set(df_flagged["Text ID"].tolist())
    true_error = df_all["Error Flag"].astype(str) == "1"
    pred_error = df_all["Text ID"].isin(predicted_error_ids)
    filter_acc = (pred_error == true_error).mean()
    print(f"  LOTUS: Subtask A error-flag accuracy {filter_acc:.4f} over {len(df_all)} notes")

    if "Error Sentence ID" in df_id.columns:
        id_acc = (
            df_id["_error_sentence_id"].astype(str).str.strip()
            == df_id["Error Sentence ID"].astype(str).str.strip()
        ).mean()
        print(f"  LOTUS: Subtask B sentence-id accuracy {id_acc:.4f} over {len(df_id)} flagged notes")

    if "Corrected Sentence" in df_correction.columns:
        corr_acc = (
            df_correction["_corrected_sentence"].astype(str).str.strip()
            == df_correction["Corrected Sentence"].astype(str).str.strip()
        ).mean()
        print(
            f"  LOTUS: Subtask C exact corrected-sentence accuracy "
            f"{corr_acc:.4f} over {len(df_correction)} flagged notes"
        )


def main():
    df = load_medec(MEDEC_CSV)
    log = []
    t0 = time.time()
    print(f"  LOTUS: loaded {len(df)} MEDEC notes from {MEDEC_CSV}")
    if "Error Flag" in df.columns:
        gold_error_rate = (df["Error Flag"].astype(str) == "1").mean()
        print(f"  LOTUS: gold error rate {gold_error_rate:.4f}")

    configure_lotus(FILTER_MAX_TOKENS, log)
    df_filter = df.sem_filter(scenarios.MEDEC_ERROR_FILTER)
    print(f"  LOTUS: filter {len(df_filter)}/{len(df)} flagged ({time.time() - t0:.1f}s)")
    print(f"  LOTUS: predicted error rate {len(df_filter) / len(df):.4f}")

    configure_lotus(MAX_TOKENS, log)
    df_id = df_filter.sem_map(
        scenarios.MEDEC_ERROR_SENTENCE_ID_MAP,
        suffix="_error_sentence_id",
    )
    print(f"  LOTUS: map sentence ids for {len(df_id)} notes ({time.time() - t0:.1f}s)")

    df_correction = df_id.sem_map(
        scenarios.MEDEC_CORRECTED_SENTENCE_MAP,
        suffix="_corrected_sentence",
    )
    print(f"  LOTUS: map corrections for {len(df_correction)} notes ({time.time() - t0:.1f}s)")

    print_eval(df, df_filter, df_id, df_correction)
    print(df_correction[["Text ID", "_error_sentence_id", "_corrected_sentence"]].head(20))

    rows = []
    filter_len = len(df)
    id_len = len(df_filter)
    correction_len = len(df_id)
    for i in range(0, min(filter_len, len(log))):
        rows.append({"op": "filter", "lotus_input": log[i]["input"], "lotus_output": log[i]["output"]})
    for i in range(filter_len, min(filter_len + id_len, len(log))):
        rows.append({"op": "map_sentence_id", "lotus_input": log[i]["input"], "lotus_output": log[i]["output"]})
    start = filter_len + id_len
    for i in range(start, min(start + correction_len, len(log))):
        rows.append({"op": "map_correction", "lotus_input": log[i]["input"], "lotus_output": log[i]["output"]})

    output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
    write_csv(output_csv, rows)
    print(f"  Saved {output_csv}")


if __name__ == "__main__":
    main()
