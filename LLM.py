from typing import Any

from langchain.agents import create_agent
from langchain_ollama import ChatOllama


class TutorLLM:
    def __init__(
        self,
        model_name: str = "gemma:7b",
        temperature: float = 0.4,
    ) -> None:
        self.model = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        self.agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt="You are a friendly English tutor. Keep responses concise.",
        )

    def respond(self, user_text: str) -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_text}]}
        )
        return self._extract_text(result)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            for msg in reversed(messages):
                msg_type = getattr(msg, "type", None)
                role = getattr(msg, "role", None)
                if msg_type == "ai" or role == "assistant":
                    return self._content_to_str(getattr(msg, "content", ""))
        return "I had trouble generating a response. Please repeat."

    def _content_to_str(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return " ".join(part for part in parts if part).strip()
        return str(content)
