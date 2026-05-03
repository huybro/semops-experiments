import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from cli_utils import parse_query_args
import scenarios


CSV_FIELDS = [
    "timestamp_utc",
    "run_id",
    "query_index",
    "total_queries",
    "endpoint",
    "model_name",
    "data_path",
    "num_ops",
    "op_chain",
    "op_prompts",
    "client_latency_sec",
    "server_latency_sec",
    "num_output_rows",
    "request_id",
    "predicate_result",
    "response_keys",
    "http_status",
]


def _truncate(value: Any, max_chars: int = 180) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _ops_chain(ops: list[dict[str, Any]]) -> str:
    return " > ".join(op.get("op", "unknown") for op in ops)


def _ops_prompts(ops: list[dict[str, Any]]) -> str:
    prompts = []
    for op in ops:
        args = op.get("args", {})
        prompt = args.get("prompt") or args.get("instruction")
        if prompt:
            prompts.append(_truncate(prompt, max_chars=100))
    return " || ".join(prompts)


def _results_preview(result: dict[str, Any], limit: int = 2) -> str:
    rows = result.get("results")
    if not isinstance(rows, list):
        return result
    return rows[:limit]


def _log_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "query_index": row["query_index"],
        "total_queries": row["total_queries"],
        "model_name": row["model_name"],
        "num_ops": row["num_ops"],
        "op_chain": row["op_chain"],
        "client_latency_sec": row["client_latency_sec"],
        "server_latency_sec": row["server_latency_sec"],
        "num_output_rows": row["num_output_rows"],
        "request_id": row["request_id"],
        "predicate_result": row["predicate_result"],
        "response_keys": row["response_keys"],
        "http_status": row["http_status"],
    }


def _append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


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
        print(json.dumps(payload, indent=2))

        start = time.perf_counter()

        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        end = time.perf_counter()
        elapsed = end - start

        print(f"HTTP status: {response.status_code}")
        if not response.ok:
            print("Response body preview:")
            print(_truncate(response.text, max_chars=1200))
        response.raise_for_status()

        print(f"Request latency: {elapsed:.3f} seconds")

        return response.json(), elapsed, response.status_code


if __name__ == "__main__":
    model_name, endpoint = parse_query_args()

    utc_now = datetime.utcnow()
    run_id = utc_now.strftime("%Y%m%dT%H%M%SZ")
    script_dir = Path(__file__).resolve().parent
    safe_model = _safe_slug(model_name)
    csv_path = script_dir / "logs" / f"client_arxiv_case3_{safe_model}_{run_id}.csv"

    data_path = "/home/hojaeson_umass_edu/.cache/kagglehub/datasets/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt_5000"
    query_1 = SemanticQueryBuilder(data_path, model_name=model_name).sem_filter(scenarios.CASE_3_FILTER_1)

    query_2 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
        .sem_filter(scenarios.CASE_3_FILTER_1)
        .sem_filter(scenarios.CASE_3_FILTER_2)
    )

    query_3 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
        .sem_filter(scenarios.CASE_3_FILTER_1)
        .sem_filter(scenarios.CASE_3_FILTER_2)
        .sem_filter(scenarios.CASE_3_FILTER_3)
    )

    query_4 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
        .sem_filter(scenarios.CASE_3_FILTER_1)
        .sem_filter(scenarios.CASE_3_FILTER_2)
        .sem_filter(scenarios.CASE_3_FILTER_3)
        .sem_map(scenarios.CASE_3_MAP_ARXIV)
    )
    query_5 = (
        SemanticQueryBuilder(data_path, model_name=model_name)
        .sem_map(scenarios.CASE_3_MAP_ARXIV)
        .sem_map(scenarios.CASE_3_MAP_ARXIV)
    )
    queries = [query_4]
    queries = [query_1, query_2, query_3, query_4]
    # queries = [query_5]

    print(f"Run ID: {run_id}")
    print(f"CSV output: {csv_path}")

    for index, query in enumerate(queries, start=1):
        print("\n" + "=" * 80)
        print(f"Executing query {index}/{len(queries)}")
        print("=" * 80)

        # Print JSON before sending
        payload = query.build()

        # Send request
        result, latency, http_status = query.execute(endpoint)

        num_output_rows = result.get("num_output_rows")
        request_id = result.get("request_id")
        server_latency = result.get("latency_sec")
        response_keys = ",".join(sorted(result.keys()))
        predicate_result = result.get("predicate_result")
        op_chain = _ops_chain(payload["ops"])
        op_prompts = _ops_prompts(payload["ops"])

        row = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "run_id": run_id,
            "query_index": index,
            "total_queries": len(queries),
            "endpoint": endpoint,
            "model_name": model_name,
            "data_path": payload["data_path"],
            "num_ops": len(payload["ops"]),
            "op_chain": op_chain,
            "op_prompts": op_prompts,
            "client_latency_sec": round(latency, 3),
            "server_latency_sec": server_latency,
            "num_output_rows": num_output_rows,
            "request_id": request_id,
            "predicate_result": predicate_result,
            "response_keys": response_keys,
            "http_status": http_status,
        }
        _append_csv(csv_path, row)

        print("\nQuery Summary:")
        print(json.dumps(_log_summary(row), indent=2))
        print("First 2 rows:")
        # print(json.dumps(_results_preview(result), indent=2))
        print(f"\nTotal request time: {latency:.3f} seconds")
        print(f"Logged query {index} metrics to: {csv_path}")

    print("\n" + "=" * 80)
    print(f"Completed {len(queries)} queries. CSV saved to: {csv_path}")
    print("=" * 80)
