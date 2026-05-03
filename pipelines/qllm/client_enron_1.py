import time
import json
import requests
from cli_utils import parse_query_args
import scenarios


def _response_summary(result: dict) -> dict:
    return {
        "request_id": result.get("request_id"),
        "predicate_result": result.get("predicate_result"),
        "num_output_rows": result.get("num_output_rows"),
        "latency_sec": result.get("latency_sec"),
        "response_keys": sorted(result.keys()),
    }


def _results_head(result: dict, limit: int = 2):
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
        self.plan.append({
            "op": "sem_filter",
            "args": {
                "prompt": prompt
            }
        })
        return self

    def sem_map(self, prompt: str):
        self.plan.append({
            "op": "sem_map",
            "args": {
                "prompt": prompt
            }
        })
        return self

    def sem_classify(self, classes: list[str]):
        self.plan.append({
            "op": "sem_classify",
            "args": {
                "classes": classes
            }
        })
        return self

    def sem_agg(self, instruction: str):
        self.plan.append({
            "op": "sem_agg",
            "args": {
                "instruction": instruction
            }
        })
        return self

    def sem_topk(self, instruction: str, k: int):
        self.plan.append({
            "op": "sem_topk",
            "args": {
                "instruction": instruction,
                "k": k
            }
        })
        return self

    def sem_join(self, instruction, right_table):
        self.plan.append({
            "op": "join",
            "args": {
                "instruction": instruction,
                "right_table": right_table
            }
        })
        return self

    def build(self) -> dict:
        payload = {
            "data_path": self.data_path,
            "ops": self.plan
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        return payload

    def execute(self, endpoint: str):
        payload = self.build()
        print(payload)

        start = time.perf_counter()

        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        end = time.perf_counter()
        elapsed = end - start

        response.raise_for_status()

        print(f"Request latency: {elapsed:.3f} seconds")

        return response.json(), elapsed


if __name__ == "__main__":
    model_name, endpoint = parse_query_args()

    data_path = "/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/enron-eval-number"

    query_1 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
            .sem_filter(scenarios.FILTER_ENRON_FRAUD)
    )


    query_2 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
            .sem_filter(scenarios.FILTER_ENRON_FRAUD)
            .sem_filter(scenarios.FILTER_ENRON_NOT_NEWS)
    )

    query_3 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
            .sem_filter(scenarios.FILTER_ENRON_FRAUD)
            .sem_filter(scenarios.FILTER_ENRON_NOT_NEWS)
            .sem_map(scenarios.MAP_ENRON_EXPLANATION)
    )
    queries = [query_1, query_2, query_3]
    
    for query in queries:
        # Print JSON before sending
        payload = query.build()

        # Send request
        result, latency = query.execute(endpoint)

        print("\nResponse Summary:")
        print(json.dumps(_response_summary(result), indent=2))
        print("First 2 rows:")
        print(json.dumps(_results_head(result), indent=2))
        print(f"\nTotal request time: {latency:.3f} seconds")
