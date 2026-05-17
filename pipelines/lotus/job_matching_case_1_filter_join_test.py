import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import time

from pipelines import scenarios
import lotus
from lotus.models import LM
from transformers import AutoTokenizer
from pipelines import llm_intercepter
from data_utils import write_csv, load
from pipelines.cli_utils import parse_vllm_args

project = 'lotus'
FILTER_MAX_TOKENS = 8
MAX_TOKENS = 4096
FREQUENCY_PENALTY = 0.5
REPETITION_PENALTY = 1.3
MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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


# Load Fever data
df_resume = load('/scratch/hojaeson_umass/kagglehub/snehaanbhawal/resume-dataset/versions/1/Resume/resume_txt_1', column='resume')
df_job = load('/scratch/hojaeson_umass/kagglehub/kshitizregmi/jobs-and-job-description/versions/2/job_title_des_txt_1', column='job')
# df_resume = df_resume.iloc[:20]
log = []
params = {'log': log, 'max_tokens': FILTER_MAX_TOKENS, 'tokenizer': tokenizer, 'seed': 42, 'frequency_penalty': FREQUENCY_PENALTY, 'repetition_penalty': REPETITION_PENALTY}
llm_intercepter.set_intercept(**params)

t0 = time.time()
input_len = len(df_resume)
df = df_resume.sem_filter(scenarios.RESUME_CASE_1_FILTER)
print(len(df))

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
lotus.settings.configure(lm=_lotus_lm)
params = {'log': log, 'max_tokens': MAX_TOKENS, 'tokenizer': tokenizer, 'seed': 42, 'frequency_penalty': FREQUENCY_PENALTY, 'repetition_penalty': REPETITION_PENALTY}
llm_intercepter.set_intercept(**params)

df = df.sem_join(df_job, scenarios.RESUME_CASE_1_JOIN)
# # df.sem_map(scenarios.RESUME_CASE_2_MAP)
print(len(df))
print(f"  LOTUS: {len(df_resume)}/{input_len} passed ({time.time() - t0:.1f}s)")
rows = []
for i in range(len(log)):
    rows.append({
        "lotus_input": log[i]["input"], "lotus_output": log[i]["output"],
    })

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
