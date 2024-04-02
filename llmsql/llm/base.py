import abc

from typing import Dict

DEFAULT_SYSTEM_PROMPT = """
You are a helpfule data analyst. You will receive JSON data containing various fields and their corresponding values, representing different attributes. Use these fields to provide an answer to the user query. The user query will indicate which fields to use for your response. Your response should contain only the answer and no additional formatting.
"""

class LLM(abc.ABC):
    @abc.abstractmethod
    def execute(self, fields: Dict[str, str], query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """Executes the LLM query. 

        Args:
            fields: A dict mapping from column names to values for this particular row.
            query: The user query for the LLM call. The query should specify how the column fields should be used.
            system_prompt: The system prompt to use for the LLM.

        """
        raise NotImplementedError