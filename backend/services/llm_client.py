"""
LLM Client
The service provides an interface to Google Gemini.
"""

import json
import re
import google.generativeai as genai
from backend.config import settings
from typing import Optional, Dict, Any

class LLMClient:
    """
    Wrapper for Gemini API calls.
    Provides both sync and async methods.
    """

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL
        self.temperature = settings.GEMINI_TEMPERATURE
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.total_tokens_used = 0

        # Create the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )

    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Synchronous text generation.
        """
        # Build the full prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        try:
            # Create a model with custom temperature if provided
            if temperature is not None:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
            else:
                model = self.model

            response = model.generate_content(full_prompt)
            
            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)

            return response.text
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
        
        Same parameters as generate, but returns a coroutine.
        """
        # Build the full prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nRespond with valid JSON only."

        try:
            # Create a model with custom temperature if provided
            if temperature is not None:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
            else:
                model = self.model

            response = await model.generate_content_async(full_prompt)
            
            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += getattr(response.usage_metadata, 'total_token_count', 0)

            return response.text
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
            It returns:
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
