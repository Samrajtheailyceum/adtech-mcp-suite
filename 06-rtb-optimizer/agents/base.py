"""
Base Agent

Wraps the Anthropic client with a standard agentic loop:
- system prompt injection
- tool use loop (tool_use → tool_result → continue)
- JSON extraction from free-text responses
"""

import json
import anthropic
from typing import Any, Dict, List, Optional

from config import ANTHROPIC_API_KEY, DEFAULT_MODEL


class BaseAgent:
    """Shared base for all ad tech agents."""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = DEFAULT_MODEL

    def run(self, user_message: str, tools: Optional[List[Dict]] = None) -> str:
        """Send a message and return the agent's text response."""
        messages = [{"role": "user", "content": user_message}]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self.system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # Agentic tool-use loop
        while response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(self._execute_tool(tu.name, tu.input)),
                }
                for tu in tool_uses
            ]
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages
            response = self.client.messages.create(**kwargs)

        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def structured_output(self, prompt: str) -> Dict:
        """Run agent and parse the first JSON object from the response."""
        response = self.run(prompt + "\n\nRespond with a single valid JSON object only.")
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return {"raw_response": response}

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Override in subclasses to handle tool execution."""
        return {"error": f"Tool '{tool_name}' not implemented in {self.name}."}
