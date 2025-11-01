import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenRouterLLM:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv(
            "MODEL_NAME", "openai/gpt-oss-20b"
        )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def generate(self, prompt: str, max_new_tokens=256, temperature=0.3) -> str:
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
        return response.choices[0].message.content.strip()

# Initialize with default model
llm = OpenRouterLLM()

# Simple factual query
prompt = "Hello"
answer = llm.generate(prompt, max_new_tokens=50, temperature=0.2)
print(answer)
