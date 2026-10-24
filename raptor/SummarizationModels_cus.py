"""DeepSeek summarization adapter; no model weights are loaded locally."""

import os

from openai import OpenAI

from .SummarizationModels import BaseSummarizationModel


class DeepSeekSummarizationModel(BaseSummarizationModel):
    def __init__(self, model=None, *, base_url=None, timeout=60.0,
                 max_retries=2, client=None):
        self.model = (model or os.getenv("DEEPSEEK_SUMMARIZATION_MODEL")
                      or os.getenv("DEEPSEEK_MODEL") or "deepseek-flash").strip()
        if not self.model:
            raise ValueError("summarization model must be a non-empty name")
        if client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Set DEEPSEEK_API_KEY in the environment before initialization.")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                timeout=timeout,
                max_retries=max_retries,
            )
        self.client = client

    def summarize(self, context, max_tokens=150):
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, "
                        "including as many key details as possible:\n"
                        f"{context}"
                    ),
                },
            ],
            max_tokens=max_tokens,
            stream=False,
            # Current DeepSeek models enable thinking by default. Short RAPTOR
            # summaries need the entire output budget for the summary itself.
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek returned an empty summary")
        return content.strip()
