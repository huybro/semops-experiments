import json
import os
import time
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDEC_CSV = "/home/hojaeson_umass/projects/semops-experiments/data/MEDEC-Full-TrainingSet-agreement-balanced-1000-qllm-data.csv"


def _response_summary(result: dict) -> dict:
    return {
        "request_id": result.get("request_id"),
        "predicate_result": result.get("predicate_result"),
        "num_output_rows": result.get("num_output_rows"),
        "latency_sec": result.get("latency_sec"),
        "response_keys": sorted(result.keys()),
    }


def _results_head(result: dict, limit: int = 5):
    rows = result.get("results")
    if not isinstance(rows, list):
        return result
    return rows[:limit]


class SemanticQueryBuilder:
    def __init__(self, data_path: str, model_name: str | None = None):
        self.data_path = data_path
        self.model_name = model_name
        self.plan = []

    def sem_filter(self, prompt: str):
        self.plan.append(
            {
                "op": "sem_filter",
                "args": {
                    "prompt": prompt,
                },
            }
        )
        return self

    def sem_map(self, prompt: str):
        self.plan.append(
            {
                "op": "sem_map",
                "args": {
                    "prompt": prompt,
                },
            }
        )
        return self

    def build(self) -> dict:
        payload = {
            "data_path": self.data_path,
            "ops": self.plan,
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        return payload

    def execute(self, endpoint: str):
        payload = self.build()
        print(json.dumps(payload, indent=2))

        start = time.perf_counter()
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.perf_counter() - start

        response.raise_for_status()
        print(f"Request latency: {elapsed:.3f} seconds")
        return response.json(), elapsed


if __name__ == "__main__":
    model_name, endpoint = parse_query_args()

    medec_csv = os.environ.get("MEDEC_CSV", str(DEFAULT_MEDEC_CSV))
    filter_only = os.environ.get("MEDEC_FILTER_ONLY", "0") == "1"

    query = (
        SemanticQueryBuilder(medec_csv, model_name=model_name)
        .sem_filter(scenarios.MEDEC_ERROR_FILTER)
        .sem_map(scenarios.MEDEC_ERROR_SENTENCE_ID_MAP)
        .sem_map(scenarios.MEDEC_CORRECTED_SENTENCE_MAP)
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")
