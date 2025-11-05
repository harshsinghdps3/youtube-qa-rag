import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from collections.abc import Mapping
from typing import Any
from typing_extensions import override
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


_ = load_dotenv()


class OpenRouterLLM:
    model_name: str
    client: OpenAI

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def generate(
        self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3
    ) -> str:
        """Generate answer with low temperature for factual consistency."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class LangChainLLM(LLM):
    llm: OpenRouterLLM

    @property
    @override
    def _llm_type(self) -> str:
        return "custom"

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


