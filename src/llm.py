
from collections.abc import Mapping
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing_extensions import override

_ = load_dotenv()


class OllamaLLM:
    """Use local LLM via Ollama"""

    model_name: str
    base_url: str

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.model_name = "qwen3:8b"
        self.base_url = base_url
        # Check if Ollama server is running or not
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Ollama server is not accessible")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama server is not running at {self.base_url}! "
                "First run `ollama serve`"
            )

    def generate(
        self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3
    ) -> str:
        """Generate answer via Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to generate answer from Ollama: {str(e)}")


class LangChainLLM(LLM):
    """LangChain compatible LLM wrapper"""

    llm: OllamaLLM

    @property
    @override
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return self.llm.generate(prompt, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.llm.model_name}
