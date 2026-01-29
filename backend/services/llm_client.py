"""
LLM Client
The service provides an interface to the GPT.
"""

from optparse import Option
import time
import json
from openai import OpenAI
from backend.config import settings
from typing import Optional, Dict, Any

class LLMClient:
    """
    Wrapper for API calls.
    provides both sync and async methods.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENROUTER_API_KEY)
        self.async_client = AsyncOpenAI(api_key=settings.OPENROUTER_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.total_tokens_used = 0

    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        syncrhonous text generation.
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type":"json_object"} if json_mode else {"type": "text"}
            )

            self.total_tokens_used += response.usages.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM ERROR]\t{str(e)}")
            raise

    async def agenerate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        asyncrhonous version of generate for used in FASTAPI
        
        Same parameters as generate, but returns a coroutine.
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type":"json_object"} if json_mode else {"type": "text"}
            )

            self.total_tokens_used += response.usages.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM ERROR]\t{str(e)}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate and parse json response.
        Example:
            res = llm.generate_json("Classify this query: What is AI?")
            It returns:
                {
                    "Query_type": "simple",
                    "confidence": 0.95
                }
        """

        system = system_prompt or "You are a helpful assistant that generates only valid JSON responses."

        response = self.generate(
            prompt=prompt,
            system_prompt=system,
            json_mode=True
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # FALLBACK: try to extract json from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"[JSON ERROR]\t{response}")

    async def agenerate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of generate_json
        """

        system = system_prompt or "You are a helpful assistant that generates only valid JSON responses."

        response = await self.agenerate(
            prompt=prompt,
            system_prompt=system,
            json_mode=True
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # FALLBACK: try to extract json from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"[JSON ERROR]\t{response}")

    def get_token_usage(self) -> int:
        return self.total_tokens_used

    def reset_token_usage(self) -> None:
        self.total_tokens_used = 0

# global singleton instance
_llm_client = None

def get_llm_client() -> LLMClient:
    """
    GET or CREATE the singleton instance of the LLMClient
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
