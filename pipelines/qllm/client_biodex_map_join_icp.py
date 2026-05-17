import json
import os
import time
from pathlib import Path

import requests

import scenarios
from cli_utils import parse_query_args


DEFAULT_ICP_ADDRESS = "127.0.0.1"
DEFAULT_ICP_PORT = 8080
DEFAULT_ICP_TOP_K = 17
DEFAULT_ICP_LOW_THRESHOLD = 0.85
DEFAULT_ICP_HIGH_THRESHOLD = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTICLE_DIR = PROJECT_ROOT / "data" / "articles"
DEFAULT_REACTION_DIR = PROJECT_ROOT / "data" / "reactions"


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

    def sem_join(
        self,
        instruction: str,
        right_table: str,
        *,
        icp: bool = False,
        icp_address: str = DEFAULT_ICP_ADDRESS,
        icp_port: int = DEFAULT_ICP_PORT,
        icp_top_k: int | None = DEFAULT_ICP_TOP_K,
        icp_low_threshold: float | None = None,
        icp_high_threshold: float | None = None,
    ):
        icp_args = {
            "instruction": instruction,
            "right_table": right_table,
            "icp": icp,
            "icp_address": icp_address,
            "icp_port": icp_port,
        }
        if icp_low_threshold is not None or icp_high_threshold is not None:
            icp_args["icp_low_threshold"] = icp_low_threshold
            icp_args["icp_high_threshold"] = icp_high_threshold
        else:
            icp_args["icp_top_k"] = icp_top_k

        self.plan.append(
            {
                "op": "join",
                "args": icp_args,
            }
        )
        return self

    def cartesian_product(
        self,
        right_table: str,
        *,
        icp: bool = False,
        icp_address: str = DEFAULT_ICP_ADDRESS,
        icp_port: int = DEFAULT_ICP_PORT,
        icp_top_k: int | None = DEFAULT_ICP_TOP_K,
        icp_low_threshold: float | None = None,
        icp_high_threshold: float | None = None,
    ):
        icp_args = {
            "right_table": right_table,
            "icp": icp,
            "icp_address": icp_address,
            "icp_port": icp_port,
        }
        if icp_low_threshold is not None or icp_high_threshold is not None:
            icp_args["icp_low_threshold"] = icp_low_threshold
            icp_args["icp_high_threshold"] = icp_high_threshold
        else:
            icp_args["icp_top_k"] = icp_top_k

        self.plan.append(
            {
                "op": "cp",
                "args": icp_args,
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

    article_dir = os.environ.get("BIODEX_ARTICLE_DIR", str(DEFAULT_ARTICLE_DIR))
    reaction_dir = os.environ.get("BIODEX_REACTION_DIR", str(DEFAULT_REACTION_DIR))
    icp_address = os.environ.get("BIODEX_ICP_ADDRESS", DEFAULT_ICP_ADDRESS)
    icp_port = int(os.environ.get("BIODEX_ICP_PORT", str(DEFAULT_ICP_PORT)))
    icp_low_threshold = float(os.environ.get(
        "BIODEX_ICP_LOW_THRESHOLD",
        str(DEFAULT_ICP_LOW_THRESHOLD),
    ))
    icp_high_threshold = float(os.environ.get(
        "BIODEX_ICP_HIGH_THRESHOLD",
        str(DEFAULT_ICP_HIGH_THRESHOLD),
    ))

    query = (
        SemanticQueryBuilder(article_dir, model_name=model_name)
        .sem_map(scenarios.BIODEX_MAP_REACTIONS)
        .sem_join(
            scenarios.BIODEX_JOIN_REACTION,
            reaction_dir,
            icp=True,
            icp_address=icp_address,
            icp_port=icp_port,
            icp_low_threshold=icp_low_threshold,
            icp_high_threshold=icp_high_threshold,
        )
        # .cartesian_product(
        #     reaction_dir,
        #     icp=True,
        #     icp_address=icp_address,
        #     icp_port=icp_port,
        #     icp_low_threshold=icp_low_threshold,
        #     icp_high_threshold=icp_high_threshold,
        # )
    )

    result, latency = query.execute(endpoint)

    print("\nResponse Summary:")
    print(json.dumps(_response_summary(result), indent=2))
    print(f"\nTotal request time: {latency:.3f} seconds")
