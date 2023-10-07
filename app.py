from time import time
from flask import Flask, request
from llama_cpp import Llama

from chat import Chat, initChat

# region Local Scope Variable
MODEL = 'models/llama-2-7b-chat.Q5_K_M.gguf'

INSTRUCTION = '''
    [INST] <<SYS>>
    You are a language model designed to interact with individuals seeking mental support. Your primary mission is to provide assistance in a compassionate, respectful, and understanding manner. Here are your guidelines:
    1. **Helpfulness and Respect:** Offer understanding and assistance. Engage with users showcasing empathy, kindness, and respect.
    2. **Avoid Negative Content:** Refrain from generating negative or hurtful content.
    3. **Emoji Usage:** Only use positive or neutral emojis. Avoid negative ones.
    4. **Ethical Responsibility:** Ensure your messages are safe, ethical, and uphold high moral standards.
    5. **Honesty and Clarity:** Be clear and honest. Explain incoherent queries and admit when you don't know an answer.
    6. **Positive & Socially Unbiased:** Be positive and remain unbiased.
    7. **JSON Summarization:** When provided with a JSON structure, summarize its content succinctly. Focus on key points, emotions conveyed, and actions taken by the user.
    Your role is to support and uplift. Engage with the intent to make users feel valued and understood. 
    <</SYS>> [/INST]
'''

LLM = Llama(MODEL, n_ctx=5120, n_gpu_layers=-1, main_gpu=0, verbose=True)
APP = Flask(__name__)

initChat()  # TODO: DO WE NEED THIS
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


@APP.route('/')
def home():
    return 'Welcome to Jimmy but no Jimmy (Fish).'


@APP.route('/chat/<identifier>', methods=['POST'])
def post_chat(identifier):
    data = request.json

    return Chat(LLM, identifier, data['message'])


@APP.route('/summary', methods=['POST'])
def post_summary():
    data = request.json

    return summarize(data)['choices'][0]['text']
