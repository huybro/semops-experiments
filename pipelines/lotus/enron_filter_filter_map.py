"""Pipeline: sem_filter → sem_filter → sem_map on Enron — LOTUS only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/..')

import time
import pandas as pd
import lotus
from experiment_utils_lotus import logger
from data_utils import write_csv

def load_enron(dir_path: str) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            rows.append({"filename": fname, "contents": f.read()})
    return pd.DataFrame(rows)

# Two filters (same semantics as enron-demo)
FILTER_FRAUD = (
    'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy"). '
    "Answer TRUE if it does, FALSE otherwise. Output TRUE or FALSE only.\n"
)

FILTER_NOT_NEWS = (
    "The email is not quoting from a news article or an article written by someone outside of Enron. "
    "Answer TRUE if it is NOT quoting such an article, FALSE otherwise. Output TRUE or FALSE only.\n"
)

# Map: explanation (you can change this to subject/sender if you want)
MAP_EXPLANATION = (
    "Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context."
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PIPELINE: Enron filter → filter → map (LOTUS)")
    print("=" * 60)

    df = load_enron("projects/palimpzest/testdata/enron-eval")

    logger.clear()
    t0 = time.time()

    # Filter 1
    df1 = df.sem_filter(FILTER_FRAUD)
    f1_len = len(df1)

    # Filter 2
    df2 = df1.sem_filter(FILTER_NOT_NEWS)
    f2_len = len(df2)

    # Map
    df3 = df2.sem_map(MAP_EXPLANATION, suffix="fraud_explanation")
    m_len = len(df3)

    elapsed = time.time() - t0
    print(f"  LOTUS: {len(df)}→{len(df1)}→{len(df2)} (elapsed {elapsed:.1f}s)")

    # Slice logger into stages (one LM call per row per op)
    lotus_f1_cap = logger[0:len(df)]
    lotus_f2_cap = logger[len(df):len(df) + f1_len]
    lotus_m_cap = logger[len(df) + f1_len:len(df) + f1_len + m_len]

    # Log per-stage prompts/outputs
    rows = []

    # Filter 1 logs (all emails)
    for i in range(len(df)):
        row = df.iloc[i]
        lm = lotus_f1_cap[i] if i < len(lotus_f1_cap) else {"input": "", "output": ""}
        rows.append({
            "stage": "filter1",
            "index": i,
            "filename": row["filename"],
            "lotus_input": lm["input"],
            "lotus_output": lm["output"],
        })

    # Filter 2 logs (only emails that passed filter 1)
    for i in range(f1_len):
        row = df1.iloc[i]
        lm = lotus_f2_cap[i] if i < len(lotus_f2_cap) else {"input": "", "output": ""}
        rows.append({
            "stage": "filter2",
            "index": i,
            "filename": row["filename"],
            "lotus_input": lm["input"],
            "lotus_output": lm["output"],
        })

    # Map logs (emails that passed both filters)
    for i in range(m_len):
        row = df3.iloc[i]
        lm = lotus_m_cap[i] if i < len(lotus_m_cap) else {"input": "", "output": ""}
        rows.append({
            "stage": "map",
            "index": i,
            "filename": row["filename"],
            "lotus_input": lm["input"],
            "lotus_output": lm["output"],
            "fraud_explanation": row.get("fraud_explanation", ""),
        })

    os.makedirs("logs", exist_ok=True)
    write_csv("logs/lotusenron_filter_filter_map.csv", rows)
    print("  Saved logs/lotusenron_filter_filter_map.csv")
