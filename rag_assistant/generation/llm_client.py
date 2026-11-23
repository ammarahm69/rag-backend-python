from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
import os

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Any:
        pass

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, stream: bool = False):
        if stream:
            return self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                stream=True
            )
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, api_key: str = "freellmapi-07ea736f4511f9cdc6abc278321a83de7a2429e328f9bf63",
                 base_url: str = "http://127.0.0.1:31415/v1",
                 model: str = "gemini-3-flash-preview"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, stream: bool = False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=stream
        )
        if stream:
            return response
        return response.choices[0].message.content
