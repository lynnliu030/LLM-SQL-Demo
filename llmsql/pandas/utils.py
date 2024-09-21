from typing import List
import re

from pandas import DataFrame

from llmsql.llm.base import DEFAULT_SYSTEM_PROMPT, LLM
from transformers import AutoTokenizer

def get_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

def post_http_request(
    model: str,
    prompts: List[str],
    temperature: float = 1,
    api_url: str = "http://localhost:8000/v1/completions",
    guided_choice: List[str] = None,
) -> requests.Response:
    if guided_choice:
        pload = {
            "model": model,
            "prompt": prompts,
            "temperature": temperature,
            "max_tokens": 5,
            "guided_choice": guided_choice,
        }
    else:
        pload = {
            "model": model,
            "prompt": prompts,
            "temperature": temperature,
            "max_tokens": 5,
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, data=json.dumps(pload))
    return response

def is_server_running(url="http://localhost:8000/v1/models"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            model = response.json()['data'][0]['id']
            return model
    except requests.ConnectionError:
        return None
    return None

def get_fields(user_prompt: str) -> str:
    """Get the names of all the fields specified in the user prompt."""
    pattern = r"{(.*?)}"
    return re.findall(pattern, user_prompt)

def get_field_score(df: DataFrame, field: str):
    num_distinct = df[field].nunique(dropna=True)
    avg_length = df[field].apply(lambda x: len(str(x))).mean()
    return avg_length / num_distinct

def get_ordered_columns(df: DataFrame, fields: List[str]):
    field_scores = {}

    for field in fields:
        field_scores[field] = get_field_score(df, field)
    
    reordered_fields = [field for field in sorted(fields, key=lambda field: field_scores[field], reverse=True)]

    return reordered_fields

def query(model: LLM, 
          prompt: str, 
          df: DataFrame, 
          reorder_columns: bool = True,
          reorder_rows: bool = True,
          system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          ):
    
    fields = get_fields(prompt)
    
    for field in fields:
        if field not in df.columns:
            raise ValueError(f"Provided field {field} does not exist in dataframe")
    
    if reorder_columns:
        fields = get_ordered_columns(df, fields)
        df = df[fields]
    else:
        # If reorder_columns is False, filter down to columns that appear in the prompt
        # but maintain original column order
        original_columns = df.columns
        filtered_columns = [column for column in original_columns if column in fields]
        df = df[filtered_columns]
    
    if reorder_rows:
        df = df.sort_values(by=fields)

    # Returns a list of dicts, maintaining column order.
    records = df.to_dict(orient="records")
    outputs = model.execute_batch(
        fields=records,
        query=prompt,
        system_prompt=system_prompt
    )

    return outputs
