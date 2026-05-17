import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-with-ErrorType.csv"
DEFAULT_FALSE_OUTPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-false-450-with-ErrorType.csv"
DEFAULT_FALSE_QLLM_OUTPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-false-450-qllm-data.csv"
DEFAULT_FALSE_AUDIT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-false-450-audit.csv"
DEFAULT_TRUE_OUTPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-true-550-with-ErrorType.csv"
DEFAULT_TRUE_QLLM_OUTPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-true-550-qllm-data.csv"
DEFAULT_TRUE_AUDIT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-agreement-true-550-audit.csv"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PORT = 8003
AUDIT_FIELDNAMES = ["source_idx", "Text ID", "gold_error_flag", "lotus_pred", "qllm_pred", "agree_target", "kept"]


def make_data(row: dict[str, str]) -> str:
    return (
        f"Text ID: {row.get('Text ID', '').strip()}\n\n"
        f"Clinical note:\n{row.get('Text', '').strip()}\n\n"
        f"Numbered sentences:\n{row.get('Sentences', '').strip()}"
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        tmp_path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows, _ = read_rows(path)
    return rows


def write_qllm_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        tmp_path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=["data"] + fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["data"] = make_data(row)
            writer.writerow(out)
    tmp_path.replace(path)


def write_single_csv(path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_single_qllm_csv(path: Path, row: dict[str, str], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data"] + fieldnames)
        writer.writeheader()
        out = dict(row)
        out["data"] = make_data(row)
        writer.writerow(out)


def run(cmd: list[str], env: dict[str, str]) -> str:
    completed = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout


def env_with_runtime_libs() -> dict[str, str]:
    env = os.environ.copy()
    prefix = Path(sys.prefix)
    lib_paths = [
        str(prefix / "lib"),
        str(prefix / "targets" / "x86_64-linux" / "lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        lib_paths.append(existing)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def parse_lotus_bool(output: str) -> bool:
    match = re.search(r"LOTUS: filter\s+(\d+)/1\s+flagged", output)
    if not match:
        raise ValueError(f"Could not parse LOTUS output:\n{output}")
    return int(match.group(1)) == 1


def parse_qllm_bool(output: str) -> bool:
    match = re.search(r'"num_output_rows":\s*(\d+)', output)
    if not match:
        raise ValueError(f"Could not parse QLLM output:\n{output}")
    return int(match.group(1)) == 1


def save_progress(
    output_csv: Path,
    qllm_output_csv: Path,
    audit_csv: Path,
    kept: list[dict[str, str]],
    audit: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    write_rows(output_csv, kept, fieldnames)
    write_qllm_rows(qllm_output_csv, kept, fieldnames)
    write_rows(
        audit_csv,
        audit,
        AUDIT_FIELDNAMES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan MEDEC one tuple at a time and keep rows where LOTUS and QLLM agree on the target outcome."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--target-outcome", choices=("true", "false"), default="false")
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--qllm-output-csv", default=None)
    parser.add_argument("--audit-csv", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing output/audit CSVs, skip audited source_idx values, and continue collecting rows.",
    )
    args = parser.parse_args()

    target_pred = args.target_outcome == "true"
    if args.target_count is None:
        args.target_count = 550 if target_pred else 450
    if args.output_csv is None:
        args.output_csv = str(DEFAULT_TRUE_OUTPUT if target_pred else DEFAULT_FALSE_OUTPUT)
    if args.qllm_output_csv is None:
        args.qllm_output_csv = str(DEFAULT_TRUE_QLLM_OUTPUT if target_pred else DEFAULT_FALSE_QLLM_OUTPUT)
    if args.audit_csv is None:
        args.audit_csv = str(DEFAULT_TRUE_AUDIT if target_pred else DEFAULT_FALSE_AUDIT)

    rows, fieldnames = read_rows(Path(args.input_csv))
    if args.limit > 0:
        scan_rows = list(enumerate(rows[args.start: args.start + args.limit], start=args.start))
    else:
        scan_rows = list(enumerate(rows[args.start:], start=args.start))

    output_csv = Path(args.output_csv)
    qllm_output_csv = Path(args.qllm_output_csv)
    audit_csv = Path(args.audit_csv)
    kept: list[dict[str, str]] = read_existing_rows(output_csv) if args.resume else []
    audit: list[dict[str, str]] = read_existing_rows(audit_csv) if args.resume else []
    audited_source_indices = {
        int(entry["source_idx"])
        for entry in audit
        if entry.get("source_idx", "").strip().isdigit()
    }

    if args.resume:
        print(
            f"resuming from {len(audit)} audited rows and "
            f"{len(kept)}/{args.target_count} kept rows",
            flush=True,
        )
        if kept and not audit and args.start == 0:
            raise ValueError(
                "Cannot safely infer scanned rows because the output CSV exists but the audit CSV is empty. "
                "Pass --start with the next unscanned source_idx."
            )

    lotus_script = Path(__file__).with_name("medec_filter_map_map.py")
    qllm_script = PROJECT_ROOT / "pipelines" / "qllm" / "client_medec_filter_map_map.py"

    with tempfile.TemporaryDirectory(prefix=f"medec_agreement_{args.target_outcome}_") as tmp:
        tmp_dir = Path(tmp)
        lotus_csv = tmp_dir / "one_lotus.csv"
        qllm_csv = tmp_dir / "one_qllm.csv"

        for source_idx, row in scan_rows:
            if len(kept) >= args.target_count:
                break
            if source_idx in audited_source_indices:
                continue

            write_single_csv(lotus_csv, row, fieldnames)
            write_single_qllm_csv(qllm_csv, row, fieldnames)

            lotus_env = env_with_runtime_libs()
            lotus_env["MEDEC_CSV"] = str(lotus_csv)
            lotus_env["MEDEC_FILTER_ONLY"] = "1"
            try:
                lotus_out = run(
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
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"LOTUS failed at source_idx={source_idx}, Text ID={row.get('Text ID', '')}:\n"
                    f"{exc.output or ''}"
                ) from exc
            lotus_pred = parse_lotus_bool(lotus_out)

            qllm_env = env_with_runtime_libs()
            qllm_env["MEDEC_CSV"] = str(qllm_csv)
            try:
                qllm_out = run(
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
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"QLLM failed at source_idx={source_idx}, Text ID={row.get('Text ID', '')}:\n"
                    f"{exc.output or ''}"
                ) from exc
            qllm_pred = parse_qllm_bool(qllm_out)

            agree_target = lotus_pred == target_pred and qllm_pred == target_pred
            keep = agree_target and len(kept) < args.target_count
            if keep:
                kept.append(row)

            audit.append({
                "source_idx": str(source_idx),
                "Text ID": row.get("Text ID", ""),
                "gold_error_flag": row.get("Error Flag", ""),
                "lotus_pred": "1" if lotus_pred else "0",
                "qllm_pred": "1" if qllm_pred else "0",
                "agree_target": str(agree_target),
                "kept": str(keep),
            })
            audited_source_indices.add(source_idx)

            if keep or len(audit) % args.save_every == 0:
                save_progress(output_csv, qllm_output_csv, audit_csv, kept, audit, fieldnames)

            print(
                f"idx={source_idx} id={row.get('Text ID', '')} "
                f"lotus={int(lotus_pred)} qllm={int(qllm_pred)} "
                f"kept_{args.target_outcome}={len(kept)}/{args.target_count}",
                flush=True,
            )

    save_progress(output_csv, qllm_output_csv, audit_csv, kept, audit, fieldnames)
    if len(kept) < args.target_count:
        raise ValueError(
            f"Only collected {len(kept)}/{args.target_count} agreed-{args.target_outcome} rows."
        )

    print(f"wrote {len(kept)} rows")
    print(f"output={output_csv}")
    print(f"qllm_output={qllm_output_csv}")
    print(f"audit={audit_csv}")


if __name__ == "__main__":
    main()
