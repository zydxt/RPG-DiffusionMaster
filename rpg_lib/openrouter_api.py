import requests
from typing import List, Dict
from rpg_lib.logs import logger


class OpenRouterAPI:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/Gourieff/sd-webui-rpg-diffusionmaster",
            "X-Title": "SD WebUI RPG DiffusionMaster",
        }

    def get_available_models(self) -> List[str]:
        """Fetch available models from OpenRouter API"""
        try:
            response = requests.get(f"{self.BASE_URL}/models", headers=self.headers)
            response.raise_for_status()
            models_data = response.json()

            # Extract model IDs and sort by pricing tier and name
            model_ids = [model["id"] for model in models_data["data"]]
            model_ids.sort()  # Simple sort for now, could be enhanced with custom sorting

            return model_ids
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {str(e)}")
            # Return empty list on error to gracefully handle failure
            return []

    def chat_completion(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Send chat completion request to OpenRouter API"""
        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json={"model": model, "messages": messages},
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise
