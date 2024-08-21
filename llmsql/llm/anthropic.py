import os
import json
from typing import Dict, List
from tqdm import tqdm

import anthropic
from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM

CLAUDE_MODELS = ["claude-3-5-sonnet-20240620"]
PRICING = {
    "claude-3-5-sonnet-20240620": {
        "input_tokens": (3.00/1_000_000),
        "output_tokens": (15.00/1_000_000),
        "cache_write":  (3.75/1_000_000),
        "cache_read":  (0.30/1_000_000)
    }
}

class Claude(LLM):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620", cache: bool = False):
        if model not in CLAUDE_MODELS:
            raise ValueError(f"Received {model} which is an unsupported model. Supported models are f{CLAUDE_MODELS}")
        self.model = model
        self.cache = cache
        self.api_key = api_key

        os.environ["ANTHROPIC_API_KEY"] = api_key
        self.client = anthropic.Anthropic()

    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        fields_json = json.dumps(fields)

        if not self.cache:
            user_prompt = f"Given the following data:\n\n{fields_json}\n\n answer the below query:\n\n{query}"

            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            data_prompt = f"Use the following data to answer any queries you recieve:\n\n{fields_json}"
            query_prompt = f"Use the above data to answer the following query:\n\n{query}"

            response = self.client.beta.prompt_caching.messages.create(
                model=self.model,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": data_prompt,
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": query_prompt,
                            }
                        ]
                    }
                ]
            )

        output = response.content[0].text
        cost = self._calculate_pricing(response.usage)
        return output, cost

    def execute_batch(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[str]:
        outputs = []
        costs = []
        for field in tqdm(fields):
            output, cost = self.execute(field, query, system_prompt)
            outputs.append(output)
            costs.append(cost)
        
        print("Total query cost: ", sum(costs))
        return outputs    

    def _calculate_pricing(self, usage):
        if self.model not in PRICING:
             raise ValueError(f"Received {self.model} with no pricing information. Supported models are f{PRICING.keys()}")
        
        model_pricing = PRICING[self.model]
        cost = usage.input_tokens * model_pricing["input_tokens"]
        cost += usage.output_tokens * model_pricing["output_tokens"]
        if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens is not None:
            cost += usage.cache_creation_input_tokens * model_pricing["cache_write"]
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens is not None:
            cost += usage.cache_read_input_tokens * model_pricing["cache_read"]
        return cost


