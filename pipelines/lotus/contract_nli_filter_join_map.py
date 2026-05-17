import os
import sys
import time
import threading
from contextlib import contextmanager

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import lotus
import litellm
from lotus.models import LM
from transformers import AutoTokenizer

from data_utils import load, write_csv
from pipelines import llm_intercepter, scenarios
from pipelines.cli_utils import parse_vllm_args


project = "lotus"
FILTER_MAX_TOKENS = 8
MAX_TOKENS = 8192
LITELLM_TIMEOUT = 600#3600
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3
TIMER_INTERVAL_SECONDS = int(os.environ.get("LOTUS_TIMER_INTERVAL_SECONDS", "60"))

DEFAULT_CONTRACT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "contract-nli",
    "contracts",
)
DEFAULT_HYPOTHESIS_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "contract-nli",
    "hypotheses",
)

MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# litellm._turn_on_debug()


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
        timeout=LITELLM_TIMEOUT,
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


@contextmanager
def stage_timer(stage_name, start_time, interval_seconds=TIMER_INTERVAL_SECONDS):
    stop_event = threading.Event()

    def log_elapsed():
        while not stop_event.wait(interval_seconds):
            elapsed = time.time() - start_time
            print(f"  TIMER: {stage_name} still running ({elapsed:.0f}s total)", flush=True)

    timer_thread = threading.Thread(target=log_elapsed, daemon=True)
    timer_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        timer_thread.join(timeout=1)


contract_dir = os.environ.get("CONTRACT_NLI_CONTRACT_DIR", DEFAULT_CONTRACT_DIR)
hypothesis_dir = os.environ.get("CONTRACT_NLI_HYPOTHESIS_DIR", DEFAULT_HYPOTHESIS_DIR)

df_contracts = load(contract_dir, column="contract")
df_hypotheses = load(hypothesis_dir, column="hypothesis")

log = []
t0 = time.time()
input_len = len(df_contracts)

configure_lotus(FILTER_MAX_TOKENS, log)
with stage_timer("filter", t0):
    df = df_contracts.sem_filter(scenarios.CONTRACT_NLI_VALID_CONTRACT)
print(f"  LOTUS: filter kept {len(df)}/{input_len} contracts ({time.time() - t0:.1f}s)")

with stage_timer("join", t0):
    df = df_contracts.sem_join(df_hypotheses, scenarios.CONTRACT_NLI_ENTAILMENT_JOIN)
print(f"  LOTUS: join kept {len(df)} contract/hypothesis pairs ({time.time() - t0:.1f}s)")

configure_lotus(MAX_TOKENS, log)
with stage_timer("map", t0):
    df = df.sem_map(scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT)
print(f"  LOTUS: map produced {len(df)} explanations ({time.time() - t0:.1f}s)")
print(df.head(20))

rows = []
for i in range(len(log)):
    rows.append({
        "lotus_input": log[i]["input"],
        "lotus_output": log[i]["output"],
    })

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
