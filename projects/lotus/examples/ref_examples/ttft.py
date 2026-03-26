import requests

URL = "http://localhost:8003/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
long_prefix = "context " * 800
prompt = f"{long_prefix}\nExplain briefly."


def get_prefill_metrics(metrics_url="http://localhost:8003/metrics"):
    resp = requests.get(metrics_url)
    resp.raise_for_status()

    prefill_sum = None
    prefill_count = None

    for line in resp.text.splitlines():
        if line.startswith("vllm:request_prefill_time_seconds_sum"):
            prefill_sum = float(line.split()[-1])
        elif line.startswith("vllm:request_prefill_time_seconds_count"):
            prefill_count = float(line.split()[-1])

    return prefill_sum, prefill_count


def get_avg_ttft(metrics_url="http://localhost:8003/metrics"):
    resp = requests.get(metrics_url)
    resp.raise_for_status()

    ttft_sum = None
    ttft_count = None

    for line in resp.text.splitlines():
        if line.startswith("vllm:time_to_first_token_seconds_sum"):
            ttft_sum = float(line.split()[-1])
        elif line.startswith("vllm:time_to_first_token_seconds_count"):
            ttft_count = float(line.split()[-1])

    if ttft_count is None or ttft_count == 0:
        return None

    return ttft_sum, ttft_count

# avg_ttft = get_avg_ttft()
# print("Average TTFT:", avg_ttft)
