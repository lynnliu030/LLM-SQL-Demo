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
    
    # TODO: Few-shot examples
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        fields_json = json.dumps(fields)

        user_prompt = f"Given the following data:\n {fields_json} \n answer the below query:\n"
        user_prompt += query

        if hasattr(self.tokenizer, "use_default_system_prompt") and not self.tokenizer.use_default_system_prompt:
            messages = []
        else:
            messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append({"role": "user", "content": user_prompt})

        # Construct a prompt for the chosen model given OpenAI style messages.
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )

        output = self.engine.generate(prompts=[prompt], sampling_params=self.sampling_params, use_tqdm=False)
        assert len(output) == 1
        return output[0].outputs[-1].text

    def execute_batch(self, fields: List[Dict[str, str]], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> List[str]:
        """Batched version of `execute`."""

        fields_json_list = [json.dumps(field) for field in fields]
        user_prompt = "Given the following data:\n {fields_json} \n answer the below query:\n"

        if not self.tokenizer.use_default_system_prompt:
            system_prompt_message = None
        else:
            system_prompt_message = {"role": "system", "content": system_prompt}
        
        prompts: List[str] = []
        for fields_json in fields_json_list:
            if system_prompt_message is not None:
                messages = [system_prompt_message]
            else:
                messages = []
            messages.append({"role": "user", "content": user_prompt.format(fields_json=fields_json) + query})

            prompts.append(self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            ))

        request_outputs = self.engine.generate(prompts=prompts, sampling_params=self.sampling_params)
        assert len(request_outputs) == len(fields)

        return [output.outputs[-1].text for output in request_outputs]
        



