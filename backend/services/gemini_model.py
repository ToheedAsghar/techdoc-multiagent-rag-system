"""
Gemini Model Provider
Handles all interactions with Google Gemini API using the new google-genai package.
"""

from google import genai
from google.genai import types
from backend.config import settings
from typing import Optional


class GeminiModel:
    """
    Wrapper for Google Gemini API calls using the new google-genai SDK.
    """

    def __init__(self):
        # Create client with API key
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL
        self.temperature = settings.GEMINI_TEMPERATURE
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.total_tokens_used = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Synchronous text generation using Gemini.
        """
        # Build the full prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=self.max_tokens,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )

            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)

            return response.text
        except Exception as e:
            print(f"[GEMINI ERROR]\t{str(e)}")
            raise

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Asynchronous text generation using Gemini.
        """
        # Build the full prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        try:
            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=self.max_tokens,
            )

            # Use the async client
            async_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = await async_client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )

            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)

            return response.text
        except Exception as e:
            print(f"[GEMINI ERROR]\t{str(e)}")
            raise

    def get_token_usage(self) -> int:
        return self.total_tokens_used

    def reset_token_usage(self) -> None:
        self.total_tokens_used = 0
