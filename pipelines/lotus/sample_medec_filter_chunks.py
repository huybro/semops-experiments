import argparse
import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-with-ErrorType.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "medec_filter_chunks"
DEFAULT_PREFIX = "MEDEC-Full-TrainingSet-balanced-filter-chunk"


def make_data(row: dict[str, str]) -> str:
    return (
        f"Text ID: {row.get('Text ID', '').strip()}\n\n"
        f"Clinical note:\n{row.get('Text', '').strip()}\n\n"
        f"Numbered sentences:\n{row.get('Sentences', '').strip()}"
    )


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_qllm_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data"] + fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["data"] = make_data(row)
            writer.writerow(out)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def generate(args: argparse.Namespace) -> None:
    rows, fieldnames = read_csv(Path(args.input_csv))
    errors = [r for r in rows if str(r.get("Error Flag", "")).strip() == "1"]
    non_errors = [r for r in rows if str(r.get("Error Flag", "")).strip() == "0"]

    error_per_chunk = args.chunk_size * args.error_per_100 // 100
    non_error_per_chunk = args.chunk_size - error_per_chunk
    max_chunks = min(len(errors) // error_per_chunk, len(non_errors) // non_error_per_chunk)
    num_chunks = min(args.num_chunks, max_chunks)

    if num_chunks < args.num_chunks:
        print(f"  MEDEC: requested {args.num_chunks} chunks, only making {num_chunks}.")

    rng = random.Random(args.seed)
    rng.shuffle(errors)
    rng.shuffle(non_errors)

    out_dir = Path(args.output_dir)
    manifest_rows = []
    all_rows = []
    for chunk_idx in range(num_chunks):
        start_e = chunk_idx * error_per_chunk
        start_n = chunk_idx * non_error_per_chunk
        chunk = (
            errors[start_e:start_e + error_per_chunk]
            + non_errors[start_n:start_n + non_error_per_chunk]
        )
        rng.shuffle(chunk)
        all_rows.extend(chunk)

        stem = f"{args.prefix}-{chunk_idx:03d}"
        lotus_path = out_dir / f"{stem}-with-ErrorType.csv"
        qllm_path = out_dir / f"{stem}-qllm-data.csv"
        write_rows(lotus_path, chunk, fieldnames)
        write_qllm_rows(qllm_path, chunk, fieldnames)
        manifest_rows.append({
            "chunk": str(chunk_idx),
            "rows": str(len(chunk)),
            "error_rows": str(error_per_chunk),
            "non_error_rows": str(non_error_per_chunk),
            "lotus_csv": str(lotus_path),
            "qllm_csv": str(qllm_path),
        })

    manifest_path = out_dir / f"{args.prefix}-manifest.csv"
    write_rows(
        manifest_path,
        manifest_rows,
        ["chunk", "rows", "error_rows", "non_error_rows", "lotus_csv", "qllm_csv"],
    )

    all_lotus = out_dir / f"{args.prefix}-all-{len(all_rows)}-with-ErrorType.csv"
    all_qllm = out_dir / f"{args.prefix}-all-{len(all_rows)}-qllm-data.csv"
    write_rows(all_lotus, all_rows, fieldnames)
    write_qllm_rows(all_qllm, all_rows, fieldnames)

    print(f"  MEDEC: wrote {num_chunks} chunks to {out_dir}")
    print(f"  MEDEC: chunk size={args.chunk_size}, per chunk={error_per_chunk} error/{non_error_per_chunk} non-error")
    print(f"  MEDEC: manifest={manifest_path}")
    print(f"  MEDEC: all lotus={all_lotus}")
    print(f"  MEDEC: all qllm={all_qllm}")


def merge(args: argparse.Namespace) -> None:
    manifest, _ = read_csv(Path(args.manifest_csv))
    selected = {x.strip() for x in args.chunks.split(",") if x.strip()}
    rows = []
    fieldnames = None
    for entry in manifest:
        if entry["chunk"] not in selected:
            continue
        chunk_rows, chunk_fieldnames = read_csv(Path(entry["lotus_csv"]))
        fieldnames = fieldnames or chunk_fieldnames
        rows.extend(chunk_rows)

    if args.target_rows and len(rows) != args.target_rows:
        raise ValueError(f"Selected chunks have {len(rows)} rows, expected {args.target_rows}.")

    output_csv = Path(args.output_csv)
    qllm_output_csv = Path(args.qllm_output_csv)
    write_rows(output_csv, rows, fieldnames or [])
    write_qllm_rows(qllm_output_csv, rows, fieldnames or [])
    print(f"  MEDEC: merged {len(rows)} rows")
    print(f"  MEDEC: lotus={output_csv}")
    print(f"  MEDEC: qllm={qllm_output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate")
    gen.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    gen.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    gen.add_argument("--prefix", default=DEFAULT_PREFIX)
    gen.add_argument("--chunk-size", type=int, default=50)
    gen.add_argument("--error-per-100", type=int, default=50)
    gen.add_argument("--num-chunks", type=int, default=38)
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=generate)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--manifest-csv", required=True)
    merge_parser.add_argument("--chunks", required=True, help="Comma-separated chunk ids, e.g. 0,1,2")
    merge_parser.add_argument("--target-rows", type=int, default=1000)
    merge_parser.add_argument("--output-csv", required=True)
    merge_parser.add_argument("--qllm-output-csv", required=True)
    merge_parser.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
