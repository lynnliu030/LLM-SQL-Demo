import pandas
import json
import openai
from typing import List
import re
from pandas import DataFrame
from llm.base import DEFAULT_SYSTEM_PROMPT
from llmsql import REGISTERED_MODEL

def get_fields(user_prompt: str) -> str:
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

def get_field_score(df: DataFrame, field: str):
    num_distinct = df[field].nunique(dropna=True)
    avg_length = df[field].apply(len).mean()
    return avg_length / num_distinct

def llmquery(self, user_prompt: str, df: DataFrame):
        fields_list = get_fields(user_prompt)
        field_scores = []
        for field in fields_list:
            field_scores.append(get_field_score(df, field))
        
        reordered_fields = [field for _, field in sorted(field_scores, fields_list, reverse=True)]
        client = REGISTERED_MODEL
        system_prompt = DEFAULT_SYSTEM_PROMPT

        args = {}
        responses = []
        for i in range(df[fields_list[0]].size):
            for field in fields_list:
                args[field] = df[field][i]
            responses.append(client.execute(fields=args, user_prompt=user_prompt))
        outputs = [response.choices[0].message.content for response in responses]
        return outputs