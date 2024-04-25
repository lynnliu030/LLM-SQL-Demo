import llmsql
from llmsql.llm.openai import OpenAI

import os

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please set your OpenAI API key as an environment variable.")

openai_api_key = os.environ["OPENAI_API_KEY"]
llmsql.init(OpenAI(base_url="https://api.openai.com/v1", api_key=openai_api_key))

from llmsql.duckdb import duckdb

# Create an in-memory DuckDB database
conn = duckdb.connect(database=':memory:', read_only=False)

# Create a table with 2 columns.
conn.execute("CREATE TABLE example_table (example_column INT, example_column_2 INT)")
conn.execute("INSERT INTO example_table VALUES (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)")

# Execute an LLM query to add the values in the 2 columns.
query_result = conn.execute(
    "SELECT LLM('Add {example_column} and {example_column_2}. Return just a number with no additional text.', example_column, example_column_2) from example_table").fetchall()

assert query_result == [('2',), ('4',), ('6',), ('8',), ('10',)]

