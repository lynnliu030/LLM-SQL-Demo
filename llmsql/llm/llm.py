import duckdb
import openai
import os

OPENAI_MODELS = ['gpt-3.5-turbo', 'gpt-4']

def init_credentials(model: str="gpt-3.5-turbo", api_key: str='None'):
    if model in OPENAI_MODELS:
        openai.api_key = api_key
    else:
        os.environ['HUGGINGFACE_TOKEN'] = api_key

def init_duckdb(model: str='gpt-3.5-turbo'):
    def llm(query: str, *args) -> str:
        # TODO UPDATE THIS SYSTEM PROMPT
        response_format = """
            You are a data analysis assistant who will respond with the artist name mentioned on reviews inputted. Answer with only the name.
            For instance:
            - If the review is 'I love the songs by Taylor Swift.', the answer is 'Taylor Swift'.
            - If the review is 'This album reminds me of the Beatles', the answer is 'the Beatles'.
        """
        
        num_fields = len(args)

        prompt = query

        # TODO ADD LOGIC FOR FIELD ORDERING

        for i in range(num_fields):
            field_val = args[i] if args[i] else "None"
            if isinstance(field_val, list):
                field_val = ''.join(field_val)
            prompt += args[i] + ": " + field_val + "\n"
        if len(prompt) > 16000:
            prompt = query + "N/A"

        response = openai.ChatCompletion.create(
            model=model, # engine = "deployment_name".
            messages=[
                {"role": "system", "content": response_format},
                {"role": "user", "content": prompt}
            ]
        )

        output = response['choices'][0]['message']['content']
        return output
    duckdb.create_function('llm', llm)