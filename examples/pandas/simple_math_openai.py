import llmsql
from llmsql.llm.openai import OpenAI
import llmsql.pandas

import pandas as pd
import os

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please set your OpenAI API key as an environment variable.")

openai_api_key = os.environ["OPENAI_API_KEY"]
llmsql.init(OpenAI(base_url="https://api.openai.com/v1", api_key=openai_api_key))

# Create an dummy pandas dataframe
df=pd.DataFrame({'example_column': [1, 2, 3, 4, 5], 'example_column_2': [1, 2, 3, 4, 5]})

# Execute an LLM query to add the values in the 2 columns.
query_result = df.llm_query(
    "Add {example_column} and {example_column_2}. Return just a number with no additional text.")


assert query_result == ["2", "4", "6", "8", "10"]

