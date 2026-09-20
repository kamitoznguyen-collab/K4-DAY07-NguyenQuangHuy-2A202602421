"""Các backend sinh câu trả lời cho KnowledgeBaseAgent.

`main.py` mặc định dùng `demo_llm` (giả lập, chỉ in lại prompt). Module này
cung cấp LLM thật để chấm được vế "câu trả lời của tác tử" trong rubric.
"""
from __future__ import annotations

import os

GEMINI_CHAT_MODEL = "gemini-3.6-flash"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

SYSTEM_RULE = (
    "Bạn là trợ lý tra cứu quy định học bổng đại học. "
    "CHỈ trả lời dựa trên phần Context được cung cấp. "
    "Trả lời ngắn gọn, nêu thẳng con số hoặc điều kiện được hỏi. "
    "Nếu Context không chứa thông tin, trả lời đúng một câu: 'Không có thông tin.'"
)


class GeminiChat:
    """LLM sinh câu trả lời qua Google Gemini API."""

    def __init__(self, model_name: str | None = None) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (hoặc GOOGLE_API_KEY) chưa được đặt")
        self.model_name = model_name or os.getenv("GEMINI_CHAT_MODEL", GEMINI_CHAT_MODEL)
        self._backend_name = f"gemini:{self.model_name}"
        self.client = genai.Client(api_key=api_key)

    def __call__(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=f"{SYSTEM_RULE}\n\n{prompt}",
        )
        return (resp.text or "").strip()


class DeepSeekChat:
    """LLM sinh câu trả lời qua DeepSeek (API tương thích OpenAI)."""

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY chưa được đặt")
        self.model_name = model_name or os.getenv("DEEPSEEK_CHAT_MODEL", DEEPSEEK_CHAT_MODEL)
        self._backend_name = f"deepseek:{self.model_name}"
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": SYSTEM_RULE},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()


def get_chat(provider: str | None = None):
    """Chọn backend LLM theo biến môi trường LLM_PROVIDER."""
    p = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if p == "gemini":
        return GeminiChat()
    if p == "deepseek":
        return DeepSeekChat()
    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {p}")
