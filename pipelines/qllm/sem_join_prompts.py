import pandas as pd
import re
from typing import Iterator


# -----------------------
# Text normalization
# -----------------------
def normalize(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------
# Data loaders
# -----------------------
def load_resumes(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False
    )

    df["Resume_join_text"] = (
        df["Resume_str"]
        .map(normalize)
    )

    return df


def load_jobs(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False
    )

    df["Job_join_text"] = (
        df["description"]
        .map(normalize)
    )

    return df


# -----------------------
# Prompt generation
# -----------------------
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
) -> Iterator[str]:
    for _, r in resumes_df.iloc[:n_resumes].iterrows():
        for _, j in jobs_df.iloc[:n_jobs].iterrows():
            yield make_sem_join_prompt(
                r["Resume_join_text"],
                j["Job_join_text"]
            )


def get_prompt_by_ids(resumes_df, jobs_df, resume_id, job_id):
    r = resumes_df.loc[resumes_df["ID"] == resume_id].iloc[0]
    j = jobs_df.loc[jobs_df["job_id"] == job_id].iloc[0]

    return make_sem_join_prompt(
        r["Resume_join_text"],
        j["Job_join_text"]
    )