"""DeepSeek question answering through the OpenAI-compatible API."""

import os

from openai import OpenAI

from .QAModels import BaseQAModel


class DeepSeekQAModel(BaseQAModel):
    def __init__(self, model=None, *, base_url=None, max_tokens=None,
                 timeout=60.0, max_retries=2, client=None):
        if max_tokens is None:
            try:
                max_tokens = int(os.getenv("DEEPSEEK_QA_MAX_TOKENS") or "256")
            except ValueError:
                raise ValueError("DEEPSEEK_QA_MAX_TOKENS must be a positive integer") from None
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.model = (model or os.getenv("DEEPSEEK_QA_MODEL")
                      or os.getenv("DEEPSEEK_MODEL") or "deepseek-flash").strip()
        if not self.model:
            raise ValueError("QA model must be a non-empty name")
        self.max_tokens = max_tokens
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

    def answer_question(self, context, question):
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": (
                        f"Given Context: {context}\n"
                        f"Give the best full answer to question: {question}\n"
                        "Use only the given context. If it does not contain the answer, "
                        "say that the context is insufficient."
                    ),
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek returned an empty answer")
        return content.strip()
