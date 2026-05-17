import asyncio
import aiohttp
import pandas as pd
import re
from typing import Iterator

# =========================
# Config
# =========================
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

RESUME_PATH = "/scratch/hojaeson_umass/kagglehub/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
JOB_PATH = "/scratch/hojaeson_umass/kagglehub/arshkon/linkedin-job-postings/versions/13/postings.csv"

MAX_RESUME_CHARS = 5500
MAX_JOB_CHARS = 5200

N_RESUMES = 1
N_JOBS = 20

MAX_IN_FLIGHT = 64

# =========================
# Utils
# =========================
def normalize(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# Loaders
# =========================
def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df["Resume_join_text"] = (
        df["Resume_str"]
        .map(normalize)
        .str.slice(0, MAX_RESUME_CHARS)
    )
    return df


def load_jobs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["description"])
    df["Job_join_text"] = (
        df["description"]
        .map(normalize)
        .str.slice(0, MAX_JOB_CHARS)
    )
    return df


# =========================
# Prompt generation
# =========================
def make_sem_join_prompt(resume_text: str, job_text: str) -> str:
    return (
        "You are a classifier. Answer only yes or no.\n\n"
        "Resume:\n"
        f"{resume_text}\n\n"
        "Job Description:\n"
        f"{job_text}\n\n"
        "Question:\n"
        "Is this candidate qualified for this job?"
    )


def sem_join_prompt_iter(
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    n_resumes: int,
    n_jobs: int
) -> Iterator[tuple[int, int, str]]:
    for _, r in resumes_df.iloc[:n_resumes].iterrows():
        for _, j in jobs_df.iloc[:n_jobs].iterrows():
            yield (
                int(r["ID"]),
                int(j["job_id"]),
                make_sem_join_prompt(
                    r["Resume_join_text"],
                    j["Job_join_text"]
                )
            )


# =========================
# vLLM async client
# =========================
sem = asyncio.Semaphore(MAX_IN_FLIGHT)


async def send_sem_join(session, resume_id, job_id, prompt):
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        async with session.post(VLLM_URL, json=payload) as resp:
            data = await resp.json()

            text = (
                data["choices"][0]["message"]["content"]
                .strip()
                .lower()
            )

            # # HARD GUARD
            # if text not in {"yes", "no"}:
            #     raise RuntimeError(
            #         f"Invalid output for ({resume_id}, {job_id}): {text}"
            #     )

            return resume_id, job_id, text


# =========================
# Main
# =========================
async def main():
    resumes = load_resumes(RESUME_PATH)
    jobs = load_jobs(JOB_PATH)

    # Sanity checks
    print("Resume length stats:")
    print(resumes["Resume_join_text"].str.len().iloc[100:200].describe())
    print("\nJob length stats:")
    print(jobs["Job_join_text"].str.len().iloc[100:200].describe())

    connector = aiohttp.TCPConnector(limit=MAX_IN_FLIGHT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        results = []

        for resume_id, job_id, prompt in sem_join_prompt_iter(
            resumes, jobs, N_RESUMES, N_JOBS
        ):
            tasks.append(
                asyncio.create_task(
                    send_sem_join(session, resume_id, job_id, prompt)
                )
            )

            if len(tasks) >= MAX_IN_FLIGHT * 2:
                batch = await asyncio.gather(*tasks)
                results.extend(batch)
                tasks.clear()

        if tasks:
            batch = await asyncio.gather(*tasks)
            results.extend(batch)

    print("\n=== SEM_JOIN RESULTS ===")
    for r in results:
        print(r)

def get_prompt_by_ids(resumes_df, jobs_df, resume_id, job_id):
    r = resumes_df.loc[resumes_df["ID"] == resume_id].iloc[0]
    j = jobs_df.loc[jobs_df["job_id"] == job_id].iloc[0]

    return make_sem_join_prompt(
        r["Resume_join_text"],
        j["Job_join_text"]
    )

if __name__ == "__main__":
    asyncio.run(main())
    # resumes = load_resumes(RESUME_PATH)
    # jobs = load_jobs(JOB_PATH)
    # resume_id = 11847784
    # job_id = 1829192

    # prompt = get_prompt_by_ids(resumes, jobs, resume_id, job_id)

    # print("===== PROMPT =====")
    # print(prompt)
    # print("==================")
        
