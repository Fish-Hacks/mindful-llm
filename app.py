from time import time
from flask import Flask
from llama_cpp import Llama

# region Local Scope Variable
MODEL = 'models/llama-2-7b-chat.Q5_K_M.gguf'

INSTRUCTION = '''
    [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. 
        Always answer as helpfully as possible, while being safe.  
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
        Please ensure that your responses are socially unbiased and positive in nature. 
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
        If you don't know the answer to a question, please don't share false information.
    <</SYS>> [/INST]
'''

LLM = Llama(MODEL)
# endregion


def timeit(func):
    '''
    A decorator that prints the time a function takes to execute.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function.
    '''
    def wrapper(*args, **kwargs):
        begin = time()
        result = func(*args, **kwargs)
        end = time()

        print(f'Function {func.__name__!r} executed in {(end-begin): .4f}s')

        return result
    return wrapper


@timeit
def summarize(json, **kwargs):
    '''
    Uses the LLM function to process a JSON string with a prefixed instruction and returns the result.

    Args:
        json (str): A JSON string to be processed.
        **kwargs: Arbitrary keyword arguments passed directly to the LLM function.

    Returns:
        The return value from the LLM function.

    Note:
        The actual behavior and the returned value would depend on the LLM function and the INSTRUCTION value.
        The function execution time will also be printed due to the @timeit decorator.
    '''
    return LLM(f'{INSTRUCTION} {json}', **kwargs)


output = summarize('Q: I am feeling a bit down. A: ', stop=['Q:', '\n'])
print(output)


# app = Flask(__name__)


# @app.route('/')
# def home():
#     return 'Welcome to Jimmy but no Jimmy (Fish).'
