import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)
import pandas as pd
pd.set_option('display.max_rows', None)

import time

from pipelines import scenarios
import lotus
from lotus.models import LM
from transformers import AutoTokenizer
from pipelines import llm_intercepter
from data_utils import write_csv, load_enron
from pipelines.cli_utils import parse_vllm_args


project = 'lotus'
MAX_TOKENS = 512
MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


_lotus_lm = LM(
    model=f"hosted_vllm/{MODEL_NAME}",
    api_base=VLLM_API_BASE,
    max_tokens=MAX_TOKENS,
    temperature=0,
    top_p=1,
    seed=42,
)
lotus.settings.configure(lm=_lotus_lm)


# -- LOTUS --
joined_df = load_enron(os.path.join(PROJECT_ROOT, "/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/enron-eval-number-test"), test=False)
# joined_df = joined_df.iloc[:20]
log = []
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer, 'seed': 42}
llm_intercepter.set_intercept(**params)


t0 = time.time()
df_filter1 = joined_df.sem_filter(scenarios.FILTER_ENRON_FRAUD_2)
lotus_time = time.time() - t0
print(f"{lotus_time:.1f}s")
print(df_filter1, len(df_filter1))
print(df_filter1['filename'])
# Slice logger into stages (one LM call per row per op)

lotus_f1_cap = log[0:len(joined_df)]

# -- Log (wide format: one row per original email, stage outputs as columns) --
rows = []

for i in range(len(lotus_f1_cap)):
    lm = lotus_f1_cap[i]
    rows.append({"op": 'filter_1', "lotus_input": lm["input"], "lotus_output": lm["output"]})

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
