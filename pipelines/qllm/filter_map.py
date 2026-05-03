import asyncio
import aiohttp
import pandas as pd
import re
from typing import Iterator

import json
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any
from cli_utils import parse_query_args



# =========================
# Utils
# =========================
def normalize(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False
    )

    df["data"] = (
        df["Resume_str"]
        .map(normalize)
    )

    return df

# ============================================================
# 1. Dataset abstraction (client-side only)
# ============================================================

class DataFrame:
    def __init__(self, name: str, data: List[str]):
        self.name = name
        self.data = data
        self._ops = []

    def sem_filter(self, predicate: str):
        self._ops.append({
            "type": "sem_filter",
            "predicate": predicate
        })
        return self

    def sem_map(self, instruction: str):
        self._ops.append({
            "type": "sem_map",
            "instruction": instruction
        })
        return self

    def compile(self):
        return self._ops


# ============================================================
# 2. Metadata extraction (client responsibility)
# ============================================================

def build_table_metadata(texts: List[str]) -> Dict[str, Any]:
    lengths = np.array([len(t) for t in texts], dtype=np.int64)

    return {
        "num_tuples": int(len(texts)),
        "avg_token_len": int(lengths.mean()),
        "p95_token_len": int(np.percentile(lengths, 95)),
        "max_token_len": int(lengths.max())
    }


# ============================================================
# 3. Query compiler (DSL → logical plan)
# ============================================================

def compile_query(df: DataFrame) -> Dict[str, Any]:
    operators = []
    prev = df.name

    for idx, op in enumerate(df.compile()):
        op_id = f"op{idx}"

        if op["type"] == "sem_filter":
            operators.append({
                "id": op_id,
                "type": "sem_filter",
                "input": prev,
                "predicate": op["predicate"]
            })

        elif op["type"] == "sem_map":
            operators.append({
                "id": op_id,
                "type": "sem_map",
                "input": prev,
                "instruction": op["instruction"]
            })

        prev = op_id

    return {
        "tables": {
            df.name: {
                "source": df.name
            }
        },
        "operators": operators,
        "output": prev
    }


# ============================================================
# 4. Prompt templates (stable, client-owned)
# ============================================================

def build_prompt_templates():
    return {
        "sem_filter": {
            "system": "You are a classifier. Answer only yes or no.",
            "user": (
                "Resume:\n{{tuple}}\n\n"
                "{{predicate}}\n"
                "Answer yes or no."
            )
        },
        "sem_map": {
            "system": "You are a helpful assistant.",
            "user": (
                "{{instruction}}\n\n"
                "{{tuple}}"
            )
        }
    }


# ============================================================
# 5. Build final client request
# ============================================================

def build_client_request(df: DataFrame) -> Dict[str, Any]:
    return {
        "logical_query": (
            "df.sem_filter(...).sem_map(...)"
        ),

        "logical_plan": compile_query(df),

        "table_metadata": {
            df.name: build_table_metadata(df.data)
        },

        "prompt_templates": build_prompt_templates()
    }


# ============================================================
# 6. Send to server
# ============================================================

def send_to_server(payload: Dict[str, Any], server_url: str):
    headers = {"Content-Type": "application/json"}
    resp = requests.post(
        server_url,
        headers=headers,
        data=json.dumps(payload)
    )
    resp.raise_for_status()
    return resp.json()


# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    model_name, _ = parse_query_args()

    # Load resumes (example)
    import pandas as pd

    SERVER_URL = "http://localhost:8000/semantic_query"

    RESUME_PATH = "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
    resumes = load_resumes(RESUME_PATH)


    # User-facing query
    df = DataFrame("df", resumes)

    query = (
        df
        .sem_filter("Is the candidate capable of GPU programming?")
        .sem_map("Summarize resume with skill set")
    )

    # Compile + send
    client_request = build_client_request(df)

    print("Client request payload (truncated):")
    print(json.dumps(client_request, indent=2)[:1500])

    import sys
    sys.path.append("/home/hojaeson_umass_edu/project/vllm-test/lotus/query_planner")
    from dataclasses import dataclass
    from operators.ops import sem_filter, sem_map
    from KVEstimator import KVEstimator, KVConfig
    @dataclass
    class TupleState:
        text: str
        kv_valid: bool = True

    cfg = KVConfig(
        bytes_per_token=16,
        filter_gen_tokens=2,
        map_max_gen_tokens=2048,
    )

    kv_estimator = KVEstimator(
        kv_config=cfg,
        max_kv_bytes=3 * 1024 * 1024,  # 4 GB
    )

    batch = []
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    

    import sys, os
    sys.path.insert(0, os.path.expanduser("~/project/vllm-test/vllm"))

    from vllm import LLM, SamplingParams
    import vllm
    print(vllm.__version__)
    print(vllm.__file__)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        disable_log_stats=False,
        enable_prefix_caching=True,
    )

    for text in resumes["data"]:
    
        inputs = tokenizer(text, return_tensors="pt").to('cuda')
        n_tokens = len(inputs['input_ids'][0])
        if n_tokens > 8192:
            print("Skipping resume with too many tokens:", n_tokens)
            continue
        is_true, request_id  = sem_filter(llm, text, "Is the candidate capable of GPU programming?")
        if is_true:
        # if not is_true:
            # Logically evict KV cache for this tuple
            llm.llm_engine.abort_request([request_id])
            continue

        if not is_true and kv_estimator.can_admit(n_tokens):
            kv_estimator.admit(n_tokens)
            batch.append(text)
        else:
            print("Semantic map executed on batch of size:", len(batch))
            sem_map(llm, batch, "Summarize resume with skill set")
            kv_estimator.reset()
            kv_estimator.admit(n_tokens)
            batch = [text]
