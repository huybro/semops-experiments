import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-with-ErrorType.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "MEDEC-Full-TrainingSet-balanced-sample-1000-with-ErrorType.csv"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PORT = 8003


def main():
    parser = argparse.ArgumentParser(
        description="Create a single-pass balanced MEDEC sample for Subtask A error-flag filtering."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--error-count", type=int, default=550)
    parser.add_argument("--non-error-count", type=int, default=450)
    parser.add_argument(
        "--check-70b",
        action="store_true",
        help="After sampling, run the LOTUS MEDEC filter-only check with Llama-70B.",
    )
    parser.add_argument("--check-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--check-port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--check-limit",
        type=int,
        default=0,
        help="Optional number of sampled rows to check. Use 0 to check the full sample.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    sample = []
    error_seen = 0
    non_error_seen = 0
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            flag = str(row.get("Error Flag", "")).strip()
            if flag == "1" and error_seen < args.error_count:
                sample.append(row)
                error_seen += 1
            elif flag == "0" and non_error_seen < args.non_error_count:
                sample.append(row)
                non_error_seen += 1

            if error_seen >= args.error_count and non_error_seen >= args.non_error_count:
                break

    if error_seen < args.error_count:
        raise ValueError(f"Requested {args.error_count} error rows, but only found {error_seen}.")
    if non_error_seen < args.non_error_count:
        raise ValueError(
            f"Requested {args.non_error_count} non-error rows, but only found {non_error_seen}."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)

    print(f"  MEDEC: wrote {len(sample)} rows to {output_csv}")
    print(f"  MEDEC: Error Flag=1 rows: {args.error_count}")
    print(f"  MEDEC: Error Flag=0 rows: {args.non_error_count}")
    print("  MEDEC: sampling method=single-pass original order")

    if args.check_70b:
        env = os.environ.copy()
        env["MEDEC_CSV"] = str(output_csv)
        env["MEDEC_FILTER_ONLY"] = "1"
        if args.check_limit > 0:
            env["MEDEC_LIMIT"] = str(args.check_limit)
        else:
            env.pop("MEDEC_LIMIT", None)

        check_script = Path(__file__).with_name("medec_filter_map_map.py")
        cmd = [
            sys.executable,
            str(check_script),
            "--model-name",
            args.check_model_name,
            "--port",
            str(args.check_port),
        ]
        print("  MEDEC: running 70B filter-only check")
        print("  MEDEC:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
