import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class LLMResult:
    provider: str
    text: str


class LLMClient:
    name = "base"

    def available(self) -> bool:
        return False

    async def complete_json(self, system: str, prompt: str, timeout_s: float) -> LLMResult | None:
        return None


class GroqClient(LLMClient):
    name = "groq"

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    async def complete_json(self, system: str, prompt: str, timeout_s: float) -> LLMResult | None:
        if not self.available():
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 360,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResult(self.name, text)


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    async def complete_json(self, system: str, prompt: str, timeout_s: float) -> LLMResult | None:
        if not self.available():
            return None
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 360,
                "responseMimeType": "application/json",
            },
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResult(self.name, text)


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()

    def available(self) -> bool:
        return bool(self.model)

    async def complete_json(self, system: str, prompt: str, timeout_s: float) -> LLMResult | None:
        if not self.available():
            return None
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\n{prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 360},
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return LLMResult(self.name, data.get("response", ""))


class LLMRouter:
    def __init__(self) -> None:
        clients = {
            "groq": GroqClient(),
            "gemini": GeminiClient(),
            "ollama": OllamaClient(),
        }
        order = os.getenv("LLM_PROVIDER_ORDER", "groq,gemini").lower()
        self.clients = [clients[name.strip()] for name in order.split(",") if name.strip() in clients]
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "7"))
        self.enabled = os.getenv("LLM_POLISH_ENABLED", "1").lower() not in {"0", "false", "no"}

    def active_provider_names(self) -> list[str]:
        if not self.enabled:
            return []
        return [client.name for client in self.clients if client.available()]

    async def complete_json(self, system: str, prompt: str) -> tuple[str | None, dict[str, Any] | None]:
        if not self.enabled:
            return None, None
        for client in self.clients:
            if not client.available():
                continue
            try:
                result = await client.complete_json(system, prompt, self.timeout_s)
                if not result or not result.text:
                    continue
                parsed = _parse_json_object(result.text)
                if parsed is not None:
                    return result.provider, parsed
            except Exception:
                continue
        return None, None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
