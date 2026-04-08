from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_python = Path(
        "/home/hojaeson_umass_edu/hojae_workspace/miniconda3/envs/py312/bin/python"
    )

    parser = argparse.ArgumentParser(
        description="Launch the vLLM OpenAI API server with local repo defaults."
    )
    parser.add_argument(
        "--python",
        default=str(default_python),
        help="Python interpreter used to launch vLLM.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--chat-template",
        default=str(repo_root / "llama3_chat_template.jinja"),
        help="Path to the chat template file.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory to reserve for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port for the OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for the API server.",
    )
    parser.add_argument(
        "--hf-home",
        default=str(repo_root),
        help="HF_HOME to export before launching vLLM.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(repo_root),
        help="Repository root used for cwd and PYTHONPATH setup.",
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable --enable-prefix-caching.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to vLLM. Repeat as needed.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send a warmup request after server starts.",
    )
    return parser


def warmup_request(host: str, port: int, model: str) -> None:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1,
    }

    for _ in range(60):
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code == 200:
                print("Warmup successful")
                return
        except Exception:
            pass
        time.sleep(1)

    print("Warmup failed")


def main() -> int:
    args = build_parser().parse_args()

    repo_root = Path(args.repo_root).resolve()
    python_bin = Path(args.python).expanduser()
    chat_template = Path(args.chat_template).expanduser()

    env = os.environ.copy()
    vllm_repo = str(repo_root / "vllm")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{vllm_repo}:{existing_pythonpath}" if existing_pythonpath else vllm_repo
    )
    env["HF_HOME"] = str(Path(args.hf_home).expanduser())

    cmd = [
        str(python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--chat-template",
        str(chat_template),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]

    if not args.no_prefix_caching:
        cmd.append("--enable-prefix-caching")

    cmd.extend(args.extra_arg)

    print("Launching vLLM with command:")
    print(" ".join(cmd))
    print(f"PYTHONPATH={env['PYTHONPATH']}")
    print(f"HF_HOME={env['HF_HOME']}")
    print(f"cwd={repo_root}")
    sys.stdout.flush()

    proc = subprocess.Popen(cmd, cwd=repo_root, env=env)

    if args.warmup:
        warmup_request(args.host, args.port, args.model)

    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())