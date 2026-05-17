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
    races = pd.read_csv(os.path.join(FORMULA_1_DIR, "races.csv"))
    constructor_standings = pd.read_csv(os.path.join(FORMULA_1_DIR, "constructorStandings.csv"))

    log = []
    t0 = time.time()

    configure_lotus(FILTER_MAX_TOKENS, log)
    constructors = constructors.sem_filter(
        "{constructorRef} {name} has participated in at least one Grand Prix in the 21st century."
    )
    print(f"  LOTUS case 46 select constructors: {len(constructors)} kept")
    circuits = circuits.sem_filter("{circuitRef} {name} was built after the end year of the Vietnam War.")
    print(f"  LOTUS case 46 select circuits: {len(circuits)} kept")

    circuits = circuits.rename(columns={"name": "circuit_name"})
    constructors = constructors.rename(columns={"name": "constructor_name"})
    df = pd.merge(races, circuits, on="circuitId")
    df = pd.merge(df, constructor_standings, on="raceId")
    df = pd.merge(df, constructors, on="constructorId")
    df = df[["circuit_name", "country"]].drop_duplicates()

    configure_lotus(MAX_TOKENS, log)
    df = df.sem_map(
        "{circuit_name} in {country}: return the continent of this Formula 1 circuit. "
        "Use one of Asia, Europe, North America, Oceania, South America, Africa.",
        suffix="continent",
    )
    result = df.groupby("_map").agg(circuit_count=("circuit_name", "count")).reset_index()
    print(f"  LOTUS case 46 select+select+cluster-style result ({time.time() - t0:.1f}s)")
    print(result)

    rows = [{"lotus_input": item["input"], "lotus_output": item["output"]} for item in log]
    output_csv = f"logs/{project}_formula_1_case_46_select_select_cluster.csv"
    write_csv(output_csv, rows)
    print(f"  Saved {output_csv}")


if __name__ == "__main__":
    main()
