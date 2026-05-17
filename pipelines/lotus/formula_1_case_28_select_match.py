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
from pipelines import llm_intercepter
from pipelines.cli_utils import parse_vllm_args


project = "lotus"
FILTER_MAX_TOKENS = 8
MAX_TOKENS = 4096
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3
TOKENIZER_LOCAL_FILES_ONLY = os.environ.get("TOKENIZER_LOCAL_FILES_ONLY", "1") != "0"
FORMULA_1_DIR = os.path.join(PROJECT_ROOT, "pipelines", "LROBench", "databases", "formula_1")

MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=TOKENIZER_LOCAL_FILES_ONLY)


def configure_lotus(max_tokens, log):
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


def main():
    circuits = pd.read_csv(os.path.join(FORMULA_1_DIR, "circuits.csv"))
    constructors = pd.read_csv(os.path.join(FORMULA_1_DIR, "constructors.csv"))
    circuits = circuits[["circuitId", "name", "location", "country"]].rename(
        columns={"name": "circuit_name"}
    )
    constructors = constructors[["constructorId", "constructorRef", "name", "nationality"]].rename(
        columns={"name": "constructor_name"}
    )

    log = []
    t0 = time.time()

    configure_lotus(FILTER_MAX_TOKENS, log)
    circuits = circuits.sem_filter("{country} is in Europe.")
    print(f"  LOTUS case 28 select circuits: {len(circuits)} kept")
    constructors = constructors.sem_filter("{nationality} corresponds to a European country.")
    print(f"  LOTUS case 28 select constructors: {len(constructors)} kept")

    configure_lotus(MAX_TOKENS, log)
    matched = circuits.sem_join(
        constructors,
        "{circuit_name:left} is named after the Formula 1 constructor {constructor_name:right}.",
    )
    print(f"  LOTUS case 28 select+match result ({time.time() - t0:.1f}s)")
    print(matched.head(20))

    rows = [{"lotus_input": item["input"], "lotus_output": item["output"]} for item in log]
    output_csv = f"logs/{project}_formula_1_case_28_select_match.csv"
    write_csv(output_csv, rows)
    print(f"  Saved {output_csv}")


if __name__ == "__main__":
    main()
