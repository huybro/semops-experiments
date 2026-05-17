import litellm

_original_completion = None


def set_intercept(**params):
    global _original_completion
    if _original_completion is None:
        _original_completion = litellm.completion
    log = params['log']
    MAX_TOKENS = params['max_tokens']
    tokenizer = params['tokenizer']
    seed = params.get("seed")
    timeout = params.get("timeout")
    num_retries = params.get("num_retries")
    frequency_penalty = params.get("frequency_penalty")
    repetition_penalty = params.get("repetition_penalty")

    def _interceptor(*args, **kwargs):
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        kwargs.setdefault("temperature", 0)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        if timeout is not None:
            kwargs.setdefault("timeout", timeout)
        if num_retries is not None:
            kwargs.setdefault("num_retries", num_retries)
        if frequency_penalty is not None:
            kwargs.setdefault("frequency_penalty", frequency_penalty)
        if repetition_penalty is not None:
            kwargs.setdefault("repetition_penalty", repetition_penalty)

        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            prompt_text = str(messages)

        result = _original_completion(*args, **kwargs)
        output_text = result.choices[0].message.content if result.choices else ""

        log.append({"input": prompt_text, "output": output_text})

        return result

    litellm.completion = _interceptor
