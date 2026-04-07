import argparse


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_PORT = 8003


def parse_vllm_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hosted vLLM model name, for example meta-llama/Llama-3.2-3B-Instruct.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Local vLLM OpenAI API port.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Optional full OpenAI-compatible API base. Overrides --port when provided.",
    )
    args = parser.parse_args()
    api_base = args.api_base or f"http://localhost:{args.port}/v1"
    return args.model_name, api_base
