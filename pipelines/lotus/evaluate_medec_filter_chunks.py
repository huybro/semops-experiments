import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "medec_filter_chunks"
DEFAULT_PREFIX = "MEDEC-Full-TrainingSet-balanced-filter-chunk"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PORT = 8003


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_data(row: dict[str, str]) -> str:
    return (
        f"Text ID: {row.get('Text ID', '').strip()}\n\n"
        f"Clinical note:\n{row.get('Text', '').strip()}\n\n"
        f"Numbered sentences:\n{row.get('Sentences', '').strip()}"
    )


def write_qllm_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data"] + fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["data"] = make_data(row)
            writer.writerow(out)


def run_command(cmd: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout


def parse_lotus_count(output: str) -> int:
    match = re.search(r"LOTUS: filter\s+(\d+)/(\d+)\s+flagged", output)
    if not match:
        raise ValueError(f"Could not parse LOTUS count from output:\n{output}")
    return int(match.group(1))


def parse_qllm_count(output: str) -> int:
    match = re.search(r'"num_output_rows":\s*(\d+)', output)
    if not match:
        raise ValueError(f"Could not parse QLLM count from output:\n{output}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", default=str(DEFAULT_CHUNK_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument(
        "--results-csv",
        default=str(DEFAULT_CHUNK_DIR / f"{DEFAULT_PREFIX}-eval.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-prompt-matched-sample-1000-with-ErrorType.csv"),
    )
    parser.add_argument(
        "--qllm-output-csv",
        default=str(PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-prompt-matched-sample-1000-qllm-data.csv"),
    )
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    manifest_path = chunk_dir / f"{args.prefix}-manifest.csv"
    manifest, _ = read_csv(manifest_path)
    if args.max_chunks > 0:
        manifest = manifest[: args.max_chunks]

    lotus_script = Path(__file__).with_name("medec_filter_map_map.py")
    qllm_script = PROJECT_ROOT / "pipelines" / "qllm" / "client_medec_filter_map_map.py"

    results = []
    for entry in manifest:
        chunk = entry["chunk"]
        lotus_csv = entry["lotus_csv"]
        qllm_csv = entry["qllm_csv"]

        lotus_env = os.environ.copy()
        lotus_env["MEDEC_CSV"] = lotus_csv
        lotus_env["MEDEC_FILTER_ONLY"] = "1"
        lotus_output = run_command(
            [
                sys.executable,
                str(lotus_script),
                "--model-name",
                args.model_name,
                "--port",
                str(args.port),
            ],
            lotus_env,
        )
        lotus_count = parse_lotus_count(lotus_output)

        qllm_env = os.environ.copy()
        qllm_env["MEDEC_CSV"] = qllm_csv
        qllm_output = run_command(
            [
                sys.executable,
                str(qllm_script),
                "--model-name",
                args.model_name,
                "--port",
                str(args.port),
            ],
            qllm_env,
        )
        qllm_count = parse_qllm_count(qllm_output)

        result = {
            "chunk": chunk,
            "rows": entry["rows"],
            "lotus_count": str(lotus_count),
            "qllm_count": str(qllm_count),
            "abs_diff": str(abs(lotus_count - qllm_count)),
            "lotus_csv": lotus_csv,
            "qllm_csv": qllm_csv,
        }
        results.append(result)
        print(
            f"chunk {chunk}: lotus={lotus_count}, qllm={qllm_count}, "
            f"diff={result['abs_diff']}"
        )

    results_csv = Path(args.results_csv)
    write_rows(
        results_csv,
        results,
        ["chunk", "rows", "lotus_count", "qllm_count", "abs_diff", "lotus_csv", "qllm_csv"],
    )

    rows_needed = args.target_rows
    selected_chunks_needed = rows_needed // int(results[0]["rows"])
    selected = sorted(results, key=lambda r: (int(r["abs_diff"]), int(r["chunk"])))[:selected_chunks_needed]

    merged_rows = []
    fieldnames = None
    for entry in selected:
        chunk_rows, chunk_fieldnames = read_csv(Path(entry["lotus_csv"]))
        fieldnames = fieldnames or chunk_fieldnames
        merged_rows.extend(chunk_rows)

    if len(merged_rows) != rows_needed:
        raise ValueError(f"Merged {len(merged_rows)} rows, expected {rows_needed}.")

    output_csv = Path(args.output_csv)
    qllm_output_csv = Path(args.qllm_output_csv)
    write_rows(output_csv, merged_rows, fieldnames or [])
    write_qllm_rows(qllm_output_csv, merged_rows, fieldnames or [])

    print(f"results={results_csv}")
    print("selected_chunks=" + ",".join(r["chunk"] for r in selected))
    print(
        "selected_counts="
        + json.dumps({
            "lotus": sum(int(r["lotus_count"]) for r in selected),
            "qllm": sum(int(r["qllm_count"]) for r in selected),
            "rows": len(merged_rows),
        })
    )
    print(f"lotus_output={output_csv}")
    print(f"qllm_output={qllm_output_csv}")


if __name__ == "__main__":
    main()
