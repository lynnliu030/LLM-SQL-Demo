import json
import re
import numpy as np
import sqlglot

import duckdb
from duckdb import DuckDBPyConnection

from llmsql import REGISTERED_MODEL
from llmsql.llm import LLM
TABLE = None
TABLE_SCORES = {}
ENGINE = REGISTERED_MODEL

def init(table:str, engine: LLM = None):
    global TABLE
    global TABLE_SCORES
    global ENGINE
    conn = duckdb.connect(database=':memory:', read_only=False)
    TABLE = conn.table(table)
    df = TABLE.df()
    for col in df.columns:
        avg_length = np.mean([len(x) for x in df[col]])
        unique_vals = len(TABLE.unique(col))
        TABLE_SCORES[col] = avg_length / unique_vals
    if engine:
        ENGINE = engine

def transform_table(query_cols):
    col_scores = [TABLE_SCORES[col] for col in query_cols]
    sorted_cols = [col for _, col in sorted(col_scores, query_cols, reverse=True)]
    TABLE.sort(sorted_cols)
    return sorted_cols

def llm_udf(prompt: str, contextargs: str) -> str:
    fields = json.loads(contextargs)
    output = REGISTERED_MODEL.execute(fields=fields, query=prompt)
    return output

def rewrite_sql(sql_query: str) -> str:
    """Intercepts DuckDB SQL query string and outputs an updated query."""
    conn = duckdb.connect(database=':memory:', read_only=False)

    # Define the regular expression pattern to match the LLM expression
    pattern = r"LLM\('\w+.*?', .+?\)"
    table_names = [x[1] for x in re.findall(sql_query, "((?i)from|(?i)join)\s+(?<table>\S+).+?")]
    if len(table_names) > 1:
        # TODO in progress join code, must iterate to make more robust
        # join_rule = re.findall(sql_query, "(?i)on\s+([\S\s]+)")[0]
        # for i in range(1, len(table_names)):
        #     table = table.join(conn.table(table_names[i], join_rule))
        raise ValueError("More than one table in query, only single table is supported")
    else:
        if table_names[0] != TABLE:
            raise AssertionError("More than one table in query, only single table is supported")

    # Function to transform the matched LLM expression
    def transform_match(match):
        input_str = match.group(0)
        prompt_start = input_str.find("('") + 2
        prompt_end = input_str.find("',", prompt_start)
        prompt = input_str[prompt_start:prompt_end]
        args_str = input_str[prompt_end+3:-1]  # Skip past "', " and avoid the last ")"
        args = transform_table(args_str.split(", "))
  
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

def override_connect(*args, **kwargs):
    connection = original_connect(*args, **kwargs)
    connection.create_function("LLM", llm_udf)
    return connection

def override_execute(self, sql_query: str):
    sql_query = rewrite_sql(sql_query)
    return original_execute(self, sql_query)

duckdb.connect = override_connect
duckdb.DuckDBPyConnection.execute = override_execute