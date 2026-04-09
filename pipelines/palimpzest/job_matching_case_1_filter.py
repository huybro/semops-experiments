import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."
LOTUS_ROOT = PROJECT_ROOT + "/projects/palimpzest/src"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOTUS_ROOT)


import time
import palimpzest as pz
from palimpzest.constants import Model

from pipelines import scenarios

from transformers import AutoTokenizer
from pipelines import llm_intercepter
from data_utils import write_csv, load
from pipelines.cli_utils import parse_vllm_args
from palimpzest.query.processor.config import QueryProcessorConfig

project = 'palimpzest'
MAX_TOKENS = 512
MODEL_NAME, VLLM_API_BASE = parse_vllm_args()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

PZ_MODEL = Model(f"hosted_vllm/{MODEL_NAME}")
pz_config = QueryProcessorConfig(
    api_base=VLLM_API_BASE,
    available_models=[PZ_MODEL],
    allow_model_selection=False,
    allow_bonded_query=True,
    allow_rag_reduction=False,
    allow_mixtures=False,
    allow_critic=False,
    allow_split_merge=False,
    seed=42,
    verbose=False,
    progress=False,
)

df_resume = load(
    "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/snehaanbhawal/resume-dataset/versions/1/Resume/resume_txt_20",
    column="resume",
)
df_job = load(
    "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/kshitizregmi/jobs-and-job-description/versions/2/job_title_des_txt_20",
    column="job",
)

log = []
params = {"log": log, "max_tokens": MAX_TOKENS, "tokenizer": tokenizer, "seed": 42}
llm_intercepter.set_intercept(**params)

t0 = time.time()

resume_ds = pz.MemoryDataset(id="resume-filter", vals=df_resume.to_dict("records"))
filtered_ds = resume_ds.sem_filter(
    scenarios.RESUME_CASE_1_FILTER,
    depends_on=["resume"],
)
filtered_ds = filtered_ds.run(config=pz_config).to_df()

print(filtered_ds)
pz_time = time.time() - t0
print(len(filtered_ds))
print(f"  PZ:    {len(filtered_ds)}/{len(df_resume)} passed ({pz_time:.1f}s)")

rows = []
for i in range(len(log)):
    rows.append({
        "pz_input": log[i]["input"], "pz_output": log[i]["output"],
    })

output_csv = f"logs/{project}_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
write_csv(output_csv, rows)
print(f"  Saved {output_csv}")
