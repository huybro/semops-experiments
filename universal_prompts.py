import json


def get_prompt(instruction, data, data2=None, op='sem_filter'):
    """Build a standardized prompt for any semantic operation."""

    messages = []
    system_prompt = (
        "You are a helpful assistant for executing semantic operators.\n"
        "You will be given data and an operation description.\n"
        "Apply the operation to the provided data exactly as specified and return only the required result.\n"
    )

    if system_prompt is not None:
        messages.append(
            {"role": "system", "type": "text", "content": system_prompt}
        )
    if op == 'sem_filter':
        operation = (
            "You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.\n" + \
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n" + \
            "Output TRUE or FALSE only.\n" + \
            f"Condition:{instruction}\n"
        )
    elif op == 'sem_map':
        operation = (
            "You  are presented with a context and a mapping instruction.\n"
            "Apply the instruction to the context and produce the mapped output.\n"
            "The output must strictly follow the instruction and contain no extra commentary.\n"
            f"Map Instruction:{instruction}\n"
        )
    elif op == 'sem_agg':
        operation = (
            "You are presented with multiple contexts.\n"
            "Aggregate them according to the aggregation instruction.\n"
            "The output must be a single aggregated result.\n"
            "Do not include explanations or commentary.\n"
            f"Instruction:{instruction}\n"
        )
    elif op == 'sem_join':
        operation = (
            "You are presented with two contexts.\n"
            "Determine whether the two contexts A, B together satisfy the condition.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n" + \
            "The output must strictly follow the condition and contain no extra commentary.\n"
            f"Condition:{instruction}\n"
        )

    if data2 is not None:
        user_messages = [
            {
                "role": "user",
                "type": "text",
                "content": (
                    "CONTEXT_A:\n"
                    "  {\n"
                    f"    \"text\": {data}\n"
                    "  }\n"
                    "\n\n"
                    "CONTEXT_B:\n"
                    "  {\n"
                    f"    \"text\": {data2}\n"
                    "  }\n"
                    "\n\n"
                    "TASK:\n"
                    f"{operation}\n\n"
                    "ANSWER:\n"
                ),
            }
        ]
    else:
        user_messages = [
            {
                "role": "user",
                "type": "text",
                "content": (
                    "CONTEXT:\n"
                    "  {\n"
                    f"    \"text\": {data}\n"
                    "  }\n"
                    "\n\n"
                    "TASK:\n"
                    f"{operation}\n\n"
                    "ANSWER:\n"
                ),
            }
        ]

    messages.extend(user_messages)
    return messages


def lotus_df2text_row(row_dict, cols):
    """Replicate LOTUS's df2text format: [Column]: «value» for each column."""
    return "".join(f"[{col.capitalize()}]: «{row_dict[col]}»\n" for col in cols)


def install_prompt_overrides():
    """Monkey-patch LOTUS's filter_formatter for sem_join (filter/map use *_formatter_custom directly).
    """
    import lotus.templates.task_instructions as task_instr
    from lotus.templates import prompt_utils as lotus_prompt_utils
    from lotus.templates.base import OpName

    def custom_filter_formatter(model, multimodal_data, user_instruction, *args, **kwargs):
        """Used by sem_join. Filter uses filter_formatter_custom; map uses map_formatter_custom."""
        if isinstance(multimodal_data, dict):
            data = multimodal_data.get("text", str(multimodal_data))
        else:
            data = str(multimodal_data)
        return lotus_prompt_utils.get_prompt(user_instruction, data, op=OpName.SEM_FILTER)

    task_instr.filter_formatter = custom_filter_formatter


def install_pz_prompt_overrides():

    from palimpzest.constants import Model

    # Patch __init__ to accept hosted_vllm models not in PZ's known list
    _original_init = Model.__init__
    def _patched_init(self, value, *args, **kwargs):
        if isinstance(value, str) and 'hosted_vllm' in value:
            self._value_ = value
            self.model_id = value
            self.model_name = value
            self.api_base = None
            self.vllm_kwargs = {}
            self.model_specs = {
                "provider": "hosted_vllm",
                "is_vllm_model": True,
                "is_text_model": True,
                "is_vision_model": False,
                "is_audio_model": False,
                "is_embedding_model": False,
                "is_text_image_multimodal_embedding_model": False,
                "is_o_model": False,
                "is_gpt_5_model": False,
                "is_reasoning_model": False,
                "supports_prompt_caching": False,
                "usd_per_input_token": 0.0,
                "usd_per_output_token": 0.0,
                "seconds_per_output_token": 0.0,
                "MMLU_Pro_score": 0.0,
            }
            return
        return _original_init(self, value, *args, **kwargs)
    Model.__init__ = _patched_init

    # Patch is_vllm_model to recognize hosted_vllm models
    _original_is_vllm = Model.is_vllm_model
    def _patched_is_vllm(self):
        if hasattr(self, 'value') and 'hosted_vllm' in str(self.value):
            return True
        return _original_is_vllm(self)
    Model.is_vllm_model = _patched_is_vllm
