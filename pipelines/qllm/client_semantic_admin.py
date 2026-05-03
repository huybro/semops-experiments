import argparse
import json
from typing import Any

import requests


DEFAULT_PORT = 8003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise semantic admin/debug endpoints."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port for the semantic endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional full base URL, e.g. http://localhost:8003.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--unpin-request-id",
        action="append",
        dest="unpin_request_ids",
        default=[],
        help="Request ID to send to /v1/semantic/pinned/unpin. Repeatable.",
    )
    return parser.parse_args()


def _base_url(args: argparse.Namespace) -> str:
    return (args.base_url or f"http://localhost:{args.port}").rstrip("/")


def call_endpoint(
    session: requests.Session,
    method: str,
    url: str,
    timeout: float,
    json_payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    response = session.request(method, url, json=json_payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def run_checks(
    base_url: str,
    timeout: float = 5.0,
    unpin_request_ids: list[str] | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    own_session = session is None
    session = session or requests.Session()

    endpoints = [
        ("POST", "/v1/semantic/dummy", None),
        ("GET", "/v1/semantic/healthz", None),
        ("GET", "/v1/semantic/pinned", None),
        ("GET", "/v1/semantic/scheduler_state", None),
        (
            "POST",
            "/v1/semantic/pinned/unpin",
            {"request_ids": unpin_request_ids} if unpin_request_ids else {},
        ),
    ]

    results: list[dict[str, Any]] = []
    try:
        for method, path, payload in endpoints:
            status_code, body = call_endpoint(
                session=session,
                method=method,
                url=f"{base_url}{path}",
                timeout=timeout,
                json_payload=payload,
            )
            results.append(
                {
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "body": body,
                }
            )
    finally:
        if own_session:
            session.close()

    return results


def main() -> None:
    args = parse_args()
    results = run_checks(
        base_url=_base_url(args),
        timeout=args.timeout,
        unpin_request_ids=args.unpin_request_ids,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
