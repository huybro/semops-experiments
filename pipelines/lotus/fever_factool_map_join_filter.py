import os
import json
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs, ProxyModel
from transformers import AutoTokenizer

from data_utils import write_csv
from pipelines import llm_intercepter, scenarios
from pipelines.cli_utils import parse_vllm_args
from ColbertWiki import IcpWiki, ColbertWiki


project = "lotus"
MAP_MAX_TOKENS = 64
FILTER_MAX_TOKENS = 8
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3

FEVER_CLAIMS_CSV = os.environ.get(
    "FEVER_CLAIMS_CSV",
    os.path.join(PROJECT_ROOT, "data", "fever_claims_sample_1000_data.csv"),
)
FEVER_CORPUS_CSV = os.environ.get(
    "FEVER_CORPUS_CSV",
    os.path.join(PROJECT_ROOT, "data", "beir_fever_corpus_data.csv"),
)
FEVER_LIMIT = int(os.environ.get("FEVER_LIMIT", "1000"))
FEVER_TOP_K = int(os.environ.get("FEVER_TOP_K", "5"))
COLBERT_INDEX_NAME = os.environ.get("COLBERT_INDEX_NAME", "fever_factool_wikipedia_colbert")
COLBERT_EXPERIMENT_ROOT = os.environ.get(
    "COLBERT_EXPERIMENT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "colbert_indexes"),
)
COLBERT_EXPERIMENT = os.environ.get("COLBERT_EXPERIMENT", "wikipedia")
COLBERT_COLLECTION = os.environ.get(
    "COLBERT_COLLECTION",
    os.path.join(COLBERT_EXPERIMENT_ROOT, "collections", "wikipedia.tsv"),
)
COLBERT_ROOT = os.environ.get(
    "COLBERT_ROOT",
    os.path.join(PROJECT_ROOT, "projects", "ColBERT"),
)
FEVER_SEARCH_BACKEND = os.environ.get("FEVER_SEARCH_BACKEND", "colbert")
ICP_ADDRESS = os.environ.get("ICP_ADDRESS", "127.0.0.1")
ICP_PORT = int(os.environ.get("ICP_PORT", "8080"))
ICP_CP_ID = os.environ.get("ICP_CP_ID", "fever_wikipedia")
ICP_LIMIT = int(os.environ.get("ICP_LIMIT", "10000"))
ICP_BUILD = os.environ.get("ICP_BUILD", "1") == "1"
FEVER_FILTER_MODE = os.environ.get("FEVER_FILTER_MODE", "cascade")
FEVER_HELPER_MODEL_NAME = os.environ.get("FEVER_HELPER_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
FEVER_HELPER_API_BASE = os.environ.get("FEVER_HELPER_API_BASE")
FEVER_HELPER_PORT = os.environ.get("FEVER_HELPER_PORT", "8004")
FEVER_CASCADE_RECALL_TARGET = float(os.environ.get("FEVER_CASCADE_RECALL_TARGET", "0.8"))
FEVER_CASCADE_PRECISION_TARGET = float(os.environ.get("FEVER_CASCADE_PRECISION_TARGET", "0.8"))
FEVER_CASCADE_SAMPLING_PERCENTAGE = float(os.environ.get("FEVER_CASCADE_SAMPLING_PERCENTAGE", "0.1"))
FEVER_CASCADE_FAILURE_PROBABILITY = float(os.environ.get("FEVER_CASCADE_FAILURE_PROBABILITY", "0.2"))

MODEL_NAME = None
VLLM_API_BASE = None
tokenizer = None


def optional_float_env(name, default=None):
    value = os.environ.get(name, default)
    if value is None or value == "":
        return None
    return float(value)


def configure_lotus(max_tokens, log):
    global MODEL_NAME, VLLM_API_BASE, tokenizer
    if MODEL_NAME is None or VLLM_API_BASE is None:
        MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"  LOTUS: configuring main/oracle LM {MODEL_NAME} at {VLLM_API_BASE}")
    lotus_lm = LM(
        model=f"hosted_vllm/{MODEL_NAME}",
        api_base=VLLM_API_BASE,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        seed=42,
        frequency_penalty=FREQUENCY_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
    )
    lotus.settings.configure(lm=lotus_lm)
    llm_intercepter.set_intercept(
        log=log,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        seed=42,
        frequency_penalty=FREQUENCY_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
    )


def configure_helper_lotus(max_tokens):
    helper_api_base = FEVER_HELPER_API_BASE
    if helper_api_base is None and FEVER_HELPER_PORT:
        helper_api_base = f"http://localhost:{FEVER_HELPER_PORT}/v1"
    if helper_api_base is None:
        helper_api_base = VLLM_API_BASE

    print(
        f"  LOTUS[cascade]: configuring helper/proxy LM {FEVER_HELPER_MODEL_NAME} "
        f"at {helper_api_base}"
    )

    helper_lm = LM(
        model=f"hosted_vllm/{FEVER_HELPER_MODEL_NAME}",
        api_base=helper_api_base,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        seed=42,
        frequency_penalty=FREQUENCY_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
    )
    lotus.settings.configure(helper_lm=helper_lm)


def fever_cascade_args():
    kwargs = {
        "recall_target": FEVER_CASCADE_RECALL_TARGET,
        "precision_target": FEVER_CASCADE_PRECISION_TARGET,
        "sampling_percentage": FEVER_CASCADE_SAMPLING_PERCENTAGE,
        "failure_probability": FEVER_CASCADE_FAILURE_PROBABILITY,
        "proxy_model": ProxyModel.HELPER_LM,
    }

    pos_threshold = optional_float_env("FEVER_FILTER_POS_CASCADE_THRESHOLD", "0.9")
    neg_threshold = optional_float_env("FEVER_FILTER_NEG_CASCADE_THRESHOLD", "0.7")
    if pos_threshold is not None or neg_threshold is not None:
        kwargs["filter_pos_cascade_threshold"] = pos_threshold
        kwargs["filter_neg_cascade_threshold"] = neg_threshold
        print(
            "  LOTUS[cascade]: using fixed helper thresholds "
            f"pos={pos_threshold}, neg={neg_threshold}"
        )

    helper_instruction = os.environ.get("FEVER_HELPER_FILTER_INSTRUCTION")
    if helper_instruction:
        kwargs["helper_filter_instruction"] = helper_instruction

    return CascadeArgs(**kwargs)


def run_support_filter(df_joined, mode):
    if mode == "normal":
        print("  LOTUS[normal]: running main LM filter only")
        return df_joined.sem_filter(scenarios.FEVER_FACTOOL_SUPPORT_FILTER), None

    if mode != "cascade":
        raise ValueError("FEVER_FILTER_MODE must be one of: normal, cascade, both")

    configure_helper_lotus(FILTER_MAX_TOKENS)
    print("  LOTUS[cascade]: running helper LM proxy filter before main LM fallback")
    return df_joined.sem_filter(
        scenarios.FEVER_FACTOOL_SUPPORT_FILTER,
        cascade_args=fever_cascade_args(),
        return_stats=True,
    )


def print_eval(label, df_supported, df_claims):
    if "id" in df_supported.columns and "true_label" in df_claims.columns:
        supported_ids = set(df_supported["id"].tolist())
        df_eval = df_claims.copy()
        df_eval["prediction"] = df_eval["id"].isin(supported_ids)
        if df_eval["true_label"].dtype == object:
            df_eval["true_label"] = df_eval["true_label"].map({
                True: True,
                False: False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
            })
        accuracy = (df_eval["prediction"] == df_eval["true_label"]).mean()
        print(f"  LOTUS[{label}]: merged-label accuracy {accuracy:.4f} over {len(df_eval)} claims")
    else:
        print(f"  LOTUS[{label}]: skipped accuracy because claim CSV has no id/true_label columns")


def parse_factool_queries(raw_query_output, fallback_claim):
    text = str(raw_query_output).strip()
    try:
        parsed = json.loads(text)
        queries = parsed.get("queries", [])
    except json.JSONDecodeError:
        queries = []

    if not queries:
        lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
            for line in text.splitlines()
        ]
        queries = [line for line in lines if line and "query" not in line.lower()]

    queries = [str(query).strip() for query in queries if str(query).strip()]
    if not queries:
        queries = [str(fallback_claim).strip()]
    if len(queries) == 1:
        queries.append(str(fallback_claim).strip())
    return queries[:2]


def colbert_search(df_queries):
    if FEVER_SEARCH_BACKEND == "icp":
        wiki = IcpWiki(
            collection=COLBERT_COLLECTION,
            cp_id=ICP_CP_ID,
            service_address=ICP_ADDRESS,
            service_port=ICP_PORT,
            limit=ICP_LIMIT,
            build=ICP_BUILD,
        )
    elif FEVER_SEARCH_BACKEND == "colbert":
        wiki = ColbertWiki(
            index_name=COLBERT_INDEX_NAME,
            experiment_root=COLBERT_EXPERIMENT_ROOT,
            experiment=COLBERT_EXPERIMENT,
            collection=COLBERT_COLLECTION,
            colbert_root=COLBERT_ROOT,
        )
    else:
        raise ValueError("FEVER_SEARCH_BACKEND must be one of: colbert, icp")

    rm, vs = wiki.lotus_settings()
    lotus.settings.configure(rm=rm, vs=vs)
    df_wikipedia = wiki.to_lotus_dataframe()

    rows = []
    for query_pos, raw_query_output in enumerate(df_queries["_map"].astype(str).tolist()):
        claim_row = df_queries.iloc[query_pos].to_dict()
        queries = parse_factool_queries(raw_query_output, claim_row["claim"])
        context_parts = []
        seen_pids = set()

        for query in queries:
            df_results = df_wikipedia.sem_search("content", query, K=FEVER_TOP_K)
            for pid, result in df_results.iterrows():
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                context_parts.append(result["content"])

        rows.append({
            **claim_row,
            "content": "\n\n".join(context_parts),
        })

    return pd.DataFrame(rows)



def main():
    raw_df = pd.read_csv(FEVER_CLAIMS_CSV)
    df_claims = raw_df.rename(columns={"data": "claim"}).head(FEVER_LIMIT).reset_index(drop=True)

    log = []

    t0 = time.time()
    print(
        f"  LOTUS: loaded {len(df_claims)} claims from {FEVER_CLAIMS_CSV}; "
        f"using {FEVER_SEARCH_BACKEND} search backend"
    )

    configure_lotus(MAP_MAX_TOKENS, log)
    df_mapped = df_claims.sem_map(scenarios.FEVER_FACTOOL_QUERY_MAP)
    print(f"  LOTUS: map generated {len(df_mapped)} search queries ({time.time() - t0:.1f}s)")

    df_joined = colbert_search(df_mapped).reset_index(drop=True)
    print(
        f"  ColBERT: search retrieved context for {len(df_joined)} claims "
        f"(K={FEVER_TOP_K}) ({time.time() - t0:.1f}s)"
    )

    configure_lotus(FILTER_MAX_TOKENS, log)
    filter_modes = ["normal", "cascade"] if FEVER_FILTER_MODE == "both" else [FEVER_FILTER_MODE]
    filter_outputs = {}
    filter_stats = {}
    df_supported = None

    for mode in filter_modes:
        filter_t0 = time.time()
        result = run_support_filter(df_joined, mode)
        if isinstance(result, tuple):
            df_mode_supported, stats = result
        else:
            df_mode_supported, stats = result, None

        filter_outputs[mode] = df_mode_supported
        filter_stats[mode] = stats
        df_supported = df_mode_supported
        filter_elapsed = time.time() - filter_t0
        total_elapsed = time.time() - t0
        print(
            f"  LOTUS[{mode}]: filter kept {len(df_mode_supported)}/{len(df_joined)} "
            f"supported claim/Wikipedia pairs "
            f"(filter={filter_elapsed:.1f}s, total={total_elapsed:.1f}s)"
        )
        if stats is not None:
            print(f"  LOTUS[{mode}]: cascade stats {stats}")
        print_eval(mode, df_mode_supported, df_claims)

    print(df_supported.head(20))

    rows = []
    map_len = len(df_mapped)
    filter_len = len(df_joined)
    for i in range(min(map_len, len(log))):
        rows.append({
            "op": "map",
            "lotus_input": log[i]["input"],
            "lotus_output": log[i]["output"],
        })
    for i in range(map_len, min(map_len + filter_len, len(log))):
        rows.append({
            "op": "filter",
            "lotus_input": log[i]["input"],
            "lotus_output": log[i]["output"],
        })

    output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
    write_csv(output_csv, rows)
    print(f"  Saved {output_csv}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
