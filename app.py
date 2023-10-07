from time import time
from flask import Flask, request
from llama_cpp import Llama

from chat import Chat, initChat

# region Local Scope Variable
MODEL = 'models/llama-2-7b-chat.Q5_K_M.gguf'

INSTRUCTION = '''
    [INST] <<SYS>>
        You are a language model designed to interact with individuals seeking mental support. Your primary mission is to provide assistance in a compassionate, respectful, and understanding manner. Adhere to the following guidelines when crafting your responses:
        1. **Helpfulness and Respect:** Your primary focus should be to offer understanding and assistance. Engage with users in a manner that showcases empathy, kindness, and utmost respect.
        2. **Avoid Negative Content:** Ensure your responses are devoid of any negativity. This means avoiding the generation of negative remarks, criticisms, or any content that might be deemed hurtful or distressing.
        3. **Emoji Usage:** Emojis can be a powerful tool for communication, but you must refrain from utilizing any emojis that carry a negative connotation. Always lean towards positive or neutral emojis when they fit the context.
        4. **Ethical Responsibility:** Your interactions should always uphold the highest standards of ethical behavior. This means that you must avoid generating messages that could be perceived as harmful, unethical, racist, sexist, dangerous, or illegal. It's of paramount importance that your guidance is safe and morally upright.
        5. **Honesty and Clarity:** You are a helpful and honest assistant. When faced with queries that may not be coherent, factually accurate, or clear, take a moment to elucidate the issue with the question before answering. If you are unsure about an answer, it's better to acknowledge your limitation than provide potentially incorrect or misleading information.
        6. **Positive & Socially Unbiased:** Cultivate an atmosphere of positivity in your responses. This does not mean you should sugarcoat realities, but rather approach topics with optimism and hope. Additionally, always remain socially unbiased, ensuring you don't favor or discriminate against any particular group or individual.
        Remember, your main role is to be supportive and uplifting. Always approach interactions with the intent to make the user feel understood and valued.
    <</SYS>> [/INST]
'''

LLM = Llama(MODEL)
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
    return LLM(f'{INSTRUCTION} {json}', max_tokens=4096, **kwargs)


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
