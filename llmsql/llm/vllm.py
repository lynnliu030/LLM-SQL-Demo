import asyncio
import json
from typing import Dict, Optional

from vllm import AsyncLLMEngine, EngineArgs, SamplingParams
from vllm.transformer_utils.tokenizer import get_tokenizer

from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM

class vLLM(LLM):
    def __init__(self, engine_args: EngineArgs, sampling_params: Optional[SamplingParams] = None):
        print("Starting vLLM engine...")
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        engine_model_config = self.engine.get_model_config()
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
            truncation_side="left")
        self.sampling_params = sampling_params
    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        
        fields_json = json.dumps(fields)

        user_prompt = f"Given the following data:\n {fields_json} \n answer the below query:\n"
        user_prompt += query

        messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )

        result_generator = asyncio.run(self.engine.generate(prompt, sampling_params=self.sampling_params))

        final_output = None
        for request_output in result_generator:
            final_output = request_output
            
        return final_output