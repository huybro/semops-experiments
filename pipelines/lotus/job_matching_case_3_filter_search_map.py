import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/lotus"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)

import time

from pipelines import scenarios
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS
from lotus.templates import prompt_utils
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
EMBEDDING_MODEL = os.environ.get("LOTUS_RM_MODEL", "intfloat/e5-base-v2")
JOB_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "job_title_des_txt_500_faiss_index",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

rm = SentenceTransformersRM(model=EMBEDDING_MODEL)
vs = FaissVS()

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
lotus.settings.configure(lm=_lotus_lm, rm=rm, vs=vs)


df_resume = load('/scratch/hojaeson_umass/kagglehub/snehaanbhawal/resume-dataset/versions/1/Resume/resume_txt_200', column='resume')
df_job = load('/scratch/hojaeson_umass/kagglehub/kshitizregmi/jobs-and-job-description/versions/2/job_title_des_txt_50_ultra_selective', column='job')
log = []

t0 = time.time()
input_len = len(df_resume)
job_len = len(df_job)

df_job = df_job.sem_index("job", JOB_INDEX_DIR)
print(f"  LOTUS: built job index with {job_len} rows at {JOB_INDEX_DIR} ({time.time() - t0:.1f}s)")

params = {
    'log': log,
    'max_tokens': FILTER_MAX_TOKENS,
    'tokenizer': tokenizer,
    'seed': 42,
    'frequency_penalty': FREQUENCY_PENALTY,
    'repetition_penalty': REPETITION_PENALTY,
}
llm_intercepter.set_intercept(**params)

df_filter = df_resume.sem_filter(scenarios.RESUME_CASE_3_FILTER)
if df_filter.attrs.get("index_dirs") is None:
    df_filter.attrs["index_dirs"] = {}
print(f"  LOTUS: filter {input_len}->{len(df_filter)} rows ({time.time() - t0:.1f}s)")

df_search = df_filter.sem_sim_join(
    df_job,
    left_on="resume",
    right_on="job",
    K=20,
    rsuffix="_job",
).reset_index(drop=True)
if "filename" in df_search.columns:
    df_search["resume_filename"] = df_search["filename"]

print(f"  LOTUS: sim_join {len(df_search)} pairs from {len(df_filter)} resumes and {job_len} jobs ({time.time() - t0:.1f}s)")
if not df_search.empty:
    print("  LOTUS: df_search sample row:")
    print(df_search.head(1).to_dict(orient="records")[0])

def build_map_prompt(row):
    return prompt_utils.get_data_prompt(
        str(row["resume"]),
        right_data=[str(row["job"])],
    )

df_search["prompt"] = df_search.apply(build_map_prompt, axis=1)
if not df_search.empty:
    print("  LOTUS: next sem_map prompt context:")
    print(df_search["prompt"].iloc[0])

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
params = {
    'log': log,
    'max_tokens': MAX_TOKENS,
    'tokenizer': tokenizer,
    'seed': 42,
    'frequency_penalty': FREQUENCY_PENALTY,
    'repetition_penalty': REPETITION_PENALTY,
}
llm_intercepter.set_intercept(**params)

df_map = df_search.sem_map(scenarios.RESUME_CASE_3_MAP)
print(f"  LOTUS: map {len(df_map)}/{len(df_search)} pairs ({time.time() - t0:.1f}s)")
print(df_map)

rows = []
filter_len = input_len
map_len = len(df_search)

for i in range(min(filter_len, len(log))):
    rows.append({
        "op": "filter",
        "lotus_input": log[i]["input"],
        "lotus_output": log[i]["output"],
    })

for i in range(filter_len, min(filter_len + map_len, len(log))):
    rows.append({
        "op": "map",
        "lotus_input": log[i]["input"],
        "lotus_output": log[i]["output"],
    })

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
