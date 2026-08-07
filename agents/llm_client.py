from __future__ import annotations
from typing import Any, Dict, List, Optional
from openai import AzureOpenAI, OpenAI
from config.settings import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._build_client()
        self.chat_model = (
            self.settings.azure_openai_chat_deployment
            if self.settings.llm_provider == "azure"
            else "gpt-4o-mini"
        )
        self.embedding_model = (
            self.settings.azure_openai_embedding_deployment
            if self.settings.llm_provider == "azure"
            else "text-embedding-3-small"
        )

    def _build_client(self):
        if self.settings.llm_provider == "azure":
            return AzureOpenAI(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
            )
        if not self.settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing")
        return OpenAI(api_key=self.settings.openai_api_key)

    def chat(self, messages, *, temperature=0.1, max_tokens=2048, tools=None, tool_choice=None):
        kwargs = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self._client.chat.completions.create(**kwargs)

    def chat_text(self, messages, *, temperature=0.1, max_tokens=2048) -> str:
        resp = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


_llm = None
def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm