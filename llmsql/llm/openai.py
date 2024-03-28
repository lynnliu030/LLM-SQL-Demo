import json
from typing import Dict

import openai
from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4"]

class OpenAI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-3.5-turbo"):
        if model not in OPENAI_MODELS:
            raise ValueError(f"Received {model} which is an unsupported model. Supported models are f{OPENAI_MODELS}")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        fields_json = json.dumps(fields)

        user_prompt = f"Given the following data:\n {fields_json} \n answer the below query:\n"
        user_prompt += query

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        output = response.choices[0].message.content
        return output


