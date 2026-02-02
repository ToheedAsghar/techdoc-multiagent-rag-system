"""
LLM Client
Unified interface for LLM providers (Gemini and GPT).
Automatically selects provider based on configuration.
"""

import json
import re
from backend.config import settings
from typing import Optional, Dict, Any, Union

from backend.services.gemini_model import GeminiModel
from backend.services.gpt_model import GPTModel


class LLMClient:
    """
    Unified wrapper for LLM API calls.
    Supports both Gemini and GPT providers based on configuration.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: Override the default provider. Options: "gemini" or "gpt".
                     If None, uses the LLM_PROVIDER setting.
        """
        self.provider = provider or settings.LLM_PROVIDER
        self.total_tokens_used = 0

        # Initialize the appropriate model based on provider
        if self.provider.lower() == "gemini":
            self._model = GeminiModel()
        elif self.provider.lower() == "gpt":
            self._model = GPTModel()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}. Use 'gemini' or 'gpt'.")

        print(f"[LLM CLIENT]\tInitialized with provider: {self.provider}")

    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Synchronous text generation.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Override temperature setting
            json_mode: If True, requests JSON-only response
            
        Returns:
            Generated text response
        """
        try:
            response = self._model.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=json_mode
            )
            self.total_tokens_used = self._model.get_token_usage()
            return response
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
        Asynchronous version of generate for use in FastAPI.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Override temperature setting
            json_mode: If True, requests JSON-only response
            
        Returns:
            Generated text response
        """
        try:
            response = await self._model.agenerate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=json_mode
            )
            self.total_tokens_used = self._model.get_token_usage()
            return response
        except Exception as e:
            print(f"[LLM ERROR]\t{str(e)}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.
        
        Example:
            res = llm.generate_json("Classify this query: What is AI?")
            Returns:
                {
                    "query_type": "simple",
                    "confidence": 0.95
                }
        """
        system = system_prompt or "You are a helpful assistant that generates only valid JSON responses."

        response = self.generate(
            prompt=prompt,
            system_prompt=system,
            json_mode=True
        )

        return self._parse_json_response(response)

    async def agenerate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of generate_json.
        """
        system = system_prompt or "You are a helpful assistant that generates only valid JSON responses."

        response = await self.agenerate(
            prompt=prompt,
            system_prompt=system,
            json_mode=True
        )

        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # FALLBACK: try to extract json from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"[JSON ERROR]\t{response}")

    def get_token_usage(self) -> int:
        """Get total tokens used across all calls."""
        return self._model.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset the token usage counter."""
        self._model.reset_token_usage()
        self.total_tokens_used = 0

    def get_provider(self) -> str:
        """Get the current provider name."""
        return self.provider


# Global singleton instances (one per provider)
_llm_clients: Dict[str, LLMClient] = {}


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Get or create the LLM client instance.
    
    Args:
        provider: Optional provider override ("gemini" or "gpt").
                 If None, uses the LLM_PROVIDER setting.
    
    Returns:
        LLMClient instance for the specified provider.
    """
    global _llm_clients
    
    # Determine which provider to use
    provider_key = (provider or settings.LLM_PROVIDER).lower()
    
    if provider_key not in _llm_clients:
        _llm_clients[provider_key] = LLMClient(provider=provider_key)
    
    return _llm_clients[provider_key]


def get_gemini_client() -> LLMClient:
    """
    Convenience function to get the Gemini client.
    """
    return get_llm_client(provider="gemini")


def get_gpt_client() -> LLMClient:
    """
    Convenience function to get the GPT client.
    """
    return get_llm_client(provider="gpt")
