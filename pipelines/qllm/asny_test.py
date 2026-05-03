


import sys, os
sys.path.insert(0, os.path.expanduser("~/project/vllm-test/vllm"))
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams
from vllm.engine.protocol import EngineCoreRequest
from vllm.inputs import TokensPrompt
import asyncio
import time
from typing import List

from transformers import AutoTokenizer


MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def make_engine_request(
    request_id: str,
    token_ids: List[int],
    sampling_params: SamplingParams,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=None,
        # client_index/current_wave/priority/trace_headers omitted (defaults)
    )

async def run_one(async_llm, idx: int):
    request_id = f"req-{idx}"

    prompt = f"Explain GPU programming basics. Request {idx}."

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
    )

    gen = async_llm.generate(
        prompt,              # ← PromptType (string)
        sampling_params,     # ← SamplingParams
        request_id,          # ← request_id
    )


    out_text = []
    async for out in gen:
        if out.outputs:
            print(out.outputs[0].text, end="", flush=True)
            out_text.append(out.outputs[0].text)

    return request_id, "".join(out_text)


async def main():
    engine_args = AsyncEngineArgs(
        model=MODEL,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    async_llm = AsyncLLM.from_engine_args(engine_args)

    try:
        tasks = [asyncio.create_task(run_one(async_llm, i)) for i in range(4)]
        results = await asyncio.gather(*tasks)

        for rid, text in results:
            print(f"\n=== {rid} ===\n{text}\n")

    finally:
        async_llm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
