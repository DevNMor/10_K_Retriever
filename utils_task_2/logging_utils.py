# utils_task_2/logging_utils.py

import logging

def log_usage(usage, label: str):
    """
    Log the prompt/completion/total token counts under a given label.
    """
    logging.info(
        f"{label} token usage — prompt: {usage.input_tokens}, "
        f"completion: {usage.output_tokens}, total: {usage.total_tokens}"
    )
