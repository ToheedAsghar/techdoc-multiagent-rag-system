"""
GPT Model Provider
Handles all interactions with OpenAI GPT via OpenRouter.
"""

import os
from langchain_openai import ChatOpenAI
from backend.config import settings
from typing import Optional


class GPTModel:
    """
    Wrapper for OpenAI GPT API calls via OpenRouter.
    """

    def __init__(self):
        self.model_name = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.total_tokens_used = 0

        # Create the ChatOpenAI model using OpenRouter
        self.model = ChatOpenAI(
            model=f"openai/{self.model_name}",
            base_url="https://openrouter.ai/api/v1",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=settings.OPENROUTER_API_KEY
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Synchronous text generation using GPT via OpenRouter.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Build user prompt
        user_prompt = prompt
        if json_mode:
            user_prompt += "\n\nRespond with valid JSON only."
        
        messages.append(HumanMessage(content=user_prompt))

        try:
            # Create a model with custom temperature if provided
            if temperature is not None:
                model = ChatOpenAI(
                    model=f"openai/{self.model_name}",
                    base_url="https://openrouter.ai/api/v1",
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    api_key=settings.OPENROUTER_API_KEY
                )
            else:
                model = self.model

            response = model.invoke(messages)

            # Track token usage if available
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                self.total_tokens_used += usage.get('total_tokens', 0)

            return response.content
        except Exception as e:
            print(f"[GPT ERROR]\t{str(e)}")
            raise

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Asynchronous text generation using GPT via OpenRouter.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Build user prompt
        user_prompt = prompt
        if json_mode:
            user_prompt += "\n\nRespond with valid JSON only."
        
        messages.append(HumanMessage(content=user_prompt))

        try:
            # Create a model with custom temperature if provided
            if temperature is not None:
                model = ChatOpenAI(
                    model=f"openai/{self.model_name}",
                    base_url="https://openrouter.ai/api/v1",
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    api_key=settings.OPENROUTER_API_KEY
                )
            else:
                model = self.model

            response = await model.ainvoke(messages)

            # Track token usage if available
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                self.total_tokens_used += usage.get('total_tokens', 0)

            return response.content
        except Exception as e:
            print(f"[GPT ERROR]\t{str(e)}")
            raise

    def get_token_usage(self) -> int:
        return self.total_tokens_used

    def reset_token_usage(self) -> None:
        self.total_tokens_used = 0
