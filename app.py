from time import time
from flask import Flask, request
from llama_cpp import Llama

from chat import Chat, initChat
from domain.emotion import Emotion

# region Local Scope Variable
MODEL = 'models/llama-2-7b-chat.Q5_K_M.gguf'

INSTRUCTION = '''
    [INST] <<SYS>>
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

    INSTRUCTION = f'''
        [INST] <<SYS>>
            Monthly journal of my emotional state and mental wellbeing:
            {json}

            # RULES:
            - ONLY provide short and concise advices for me to improve my emotional state and mental wellbeing.
            - SHORT AND CONCISE. NO LONG PARAGRAPHS.
            - MAXIMUM 3 ADVICES.
        <</SYS>> [/INST]
    '''

    return LLM(f'{INSTRUCTION}', max_tokens=0, **kwargs)


@APP.route('/')
def home():
    return 'Welcome to Jimmy but no Jimmy (Fish).'


@APP.route('/chat/<identifier>', methods=['POST'])
def post_chat(identifier):
    data = request.json

    return Chat(LLM, identifier, data['message'])


@APP.route('/summary', methods=['POST'])
def post_summary():
    data = request.data
    data = Emotion.from_json(data)

    data = [x.emotion for x in data]
    data = ','.join(data)

    result = summarize(data)
    print(result)

    return '.'.join(result['choices'][0]['text'].split('.')[1:])
