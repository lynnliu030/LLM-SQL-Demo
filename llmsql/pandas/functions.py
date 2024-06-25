import pandas as pd
from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT
from llmsql.pandas.utils import query

@pd.api.extensions.register_dataframe_accessor("llm_query")
class LLMQuery:
    def __init__(self, pandas_obj):
        self._df = pandas_obj
    
    def __call__(self, prompt: str, reorder_columns: bool = True, reorder_rows: bool = True, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        from llmsql import REGISTERED_MODEL
        return query(model=REGISTERED_MODEL, prompt=prompt, df=self._df, reorder_columns=reorder_columns, reorder_rows=reorder_rows, system_prompt=system_prompt)