from dataclasses import asdict
import json
from typing import List, Dict, Optional

try:
    import vllm
except ImportError as e:
    raise ImportError("Run `pip install llmsql[vllm]` to use the vllm backend.")

from vllm import EngineArgs, LLM as LLMEntrypoint, SamplingParams

from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM

class vLLM(LLM):
    def __init__(self, engine_args: EngineArgs, sampling_params: Optional[SamplingParams] = None):
        print("Starting vLLM engine...")

        self.engine_args = engine_args
        self.sampling_params = sampling_params or SamplingParams()
        
        # Enable prefix caching.
        self.engine_args.enable_prefix_caching = True
        # Disable log stats by default
        self.engine_args.disable_log_stats = True

        self.engine = LLMEntrypoint(**asdict(self.engine_args))
        self.tokenizer = self.engine.get_tokenizer()

    def _generate_prompt(self, user_prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append({"role": "user", "content": user_prompt})

        successful_prompt_generation = False
        while not successful_prompt_generation:
            try:
                # Construct a prompt for the chosen model given OpenAI style messages.
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                if messages[0]["role"] == "system":
                    # Try again without system prompt
                    messages = messages[1:]
                else:
                    raise e
            else:
                successful_prompt_generation = True
        
        return prompt
    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        fields_json = json.dumps(fields)

        user_prompt = f"Given the following data:\n {fields_json} \n answer the below query:\n"
        user_prompt += query
        
        prompt = self._generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
        output = self.engine.generate(prompts=[prompt], sampling_params=self.sampling_params, use_tqdm=False)
        assert len(output) == 1
        return output[0].outputs[-1].text

    def execute_batch(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[str]:
        """Batched version of `execute`."""

        fields_json_list = [json.dumps(field) for field in fields]
        user_prompt_template = "Given the following data:\n {fields_json} \n answer the below query:\n"

        user_prompts = [user_prompt_template.format(fields_json) for fields_json in fields_json_list]

        prompts = [self._generate_prompt(user_prompt=user_prompt, system_prompt=system_prompt) for user_prompt in user_prompts]

        request_outputs = self.engine.generate(prompts=prompts, sampling_params=self.sampling_params)
        assert len(request_outputs) == len(fields)

        return [output.outputs[-1].text for output in request_outputs]
        



