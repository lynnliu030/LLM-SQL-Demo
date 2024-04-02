from llmsql.llm.base import LLM

REGISTERED_MODEL = None

def init(model_runner: LLM):
    global REGISTERED_MODEL
    REGISTERED_MODEL = model_runner