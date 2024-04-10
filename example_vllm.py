import llmsql
from llmsql.llm.vllm import vLLM

from vllm import EngineArgs

args = EngineArgs(model="openai-community/gpt2")
llm_runner = vLLM(engine_args=args)
llmsql.init(model_runner=llm_runner)

from llmsql.duckdb import duckdb

# Create an in-memory DuckDB database
conn = duckdb.connect(database=':memory:', read_only=False)

# Create a table with 2 columns.
conn.execute("CREATE TABLE example_table (example_column INT, example_column_2 INT)")
conn.execute("INSERT INTO example_table VALUES (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)")

# Execute an LLM query to add the values in the 2 columns.
query_result = conn.execute(
    "SELECT LLM('Add {example_column} and {example_column_2}. Return just a number with no additional text.', example_column, example_column_2) from example_table").fetchall()

import pdb; pdb.set_trace()

#assert query_result == [('2',), ('4',), ('6',), ('8',), ('10',)]

