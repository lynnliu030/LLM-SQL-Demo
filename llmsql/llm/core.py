import pandas
import json
import openai
from typing import List
import re
from pandas import DataFrame
from base import DEFAULT_SYSTEM_PROMPT

def rewrite_sql(sql_query: str) -> str:
    """Intercepts DuckDB SQL query string and outputs an updated query."""
    # Define the regular expression pattern to match the LLM expression
    pattern = r"{(.*?)}"

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

def get_fields(sql_query: str) -> str:
    pattern = r"{(.*?)}"
    return re.findall(pattern)

def get_field_score(df: DataFrame, field: str):
    num_distinct = df[field].nunique(dropna=True)
    avg_length = df[field].apply(len).mean()
    return avg_length / num_distinct

def query(self, query: str, df: DataFrame):
        fields_list = get_fields(query)
        field_scores = []
        for field in fields_list:
            field_scores.append(get_field_score(df, field))
        
        reordered_fields = [field for _, field in sorted(field_scores, fields_list, reverse=True)]
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        system_prompt = DEFAULT_SYSTEM_PROMPT

        args = {}
        responses = []
        for i in range(df[fields_list[0]].size):
            for field in fields_list:
                args[field] = df[field][i]
            fields_json = json.dumps(args)
            user_prompt =   query + f"Given the following data:\n {fields_json} \n answer the above query:\n"
            responses.append(client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            ))
        outputs = [response.choices[0].message.content for response in responses]
        return outputs