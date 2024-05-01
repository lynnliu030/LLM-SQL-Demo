import json
import re

import duckdb
from duckdb import DuckDBPyConnection

from llmsql import REGISTERED_MODEL

if REGISTERED_MODEL is None:
    raise RuntimeError("Call llmsql.init before importing from llmsql.duckdb")
    
def llm_udf(prompt: str, contextargs: str) -> str:
    fields = json.loads(contextargs)
    output = REGISTERED_MODEL.execute(fields=fields, query=prompt)
    return output

def rewrite_sql(sql_query: str) -> str:
    """Intercepts DuckDB SQL query string and outputs an updated query."""

    # Define the regular expression pattern to match the LLM expression
    pattern = r"LLM\('\w+.*?', .+?\)"

    # Function to transform the matched LLM expression
    def transform_match(match):
        input_str = match.group(0)
        prompt_start = input_str.find("('") + 2
        prompt_end = input_str.find("',", prompt_start)
        prompt = input_str[prompt_start:prompt_end]
        args_str = input_str[prompt_end+3:-1]  # Skip past "', " and avoid the last ")"
        args = args_str.split(", ")
        # For each value, format it as "'value', value"
        formatted_args = [f"'{val}', {val}" for val in args]
        # Join the formatted values into a single string
        json_object_str = ", ".join(formatted_args)
        json_object = f"JSON_OBJECT({json_object_str})"
        return f"LLM('{prompt}', {json_object})"
    
    transformed_llm_op = re.sub(pattern, transform_match, sql_query)
    return transformed_llm_op

# Override duckdb.sql(...)
original_sql_fn = duckdb.sql
def override_sql(sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_sql_fn(sql_query)

duckdb.sql = override_sql 
duckdb.create_function("LLM", llm_udf)


# Override duckdb.connect(...); conn.execute
original_connect = duckdb.connect
original_execute = duckdb.DuckDBPyConnection.execute
original_connection_sql = duckdb.DuckDBPyConnection.sql

def override_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connection.create_function("LLM", llm_udf)
    return connection

def override_execute(self, sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_execute(self, sql_query)

def override_connect_sql(self, sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_connection_sql(self, sql_query)

duckdb.connect = override_connect
duckdb.DuckDBPyConnection.execute = override_execute
duckdb.DuckDBPyConnection.sql = override_connect_sql