from llama_cpp import Llama
import os
import time

HISTORY = []
INSTRUCTION = '''
[INST] <<SYS>>
You are a chatbot therapist interacting with a user with a possible mental health issue.
Provide a listening ear for the user for them to express their feelings.
Respond with empathy and understanding.
<</SYS>>
[/INST]
'''

def initChat():
    HISTORY.append(f'<<SYS>> : {INSTRUCTION}')

def Chat(LLM, username = 'User', message = ''):
    HISTORY.append(f'{username} : {message}')
    
    # Convert History to String
    HISTORY_STR = ""
    for msg in HISTORY:
        HISTORY_STR += msg + "\n"

    response = LLM(HISTORY_STR + f'Q: {message}\nA: ', stop=['\n', 'Q: ', 'A: '])
    response = response['choices'][0]['text']

    HISTORY.append(f'Assistant : {response}')
    return response


if __name__ == "__main__":
    initChat()
    username = "Addison"
    llm = Llama(model_path="./models/llama-2-7b-chat.Q5_K_M.gguf", n_ctx=4096, n_gpu_layers=-1, main_gpu=0, verbose=True)

    while True:
        message = input("Message: ")
        start = time.time()
        response = Chat(llm, username, message)
        
        # Clear terminal
        os.system('cls')
        print(f"Time Taken: {time.time() - start}")

        # Print History
        for msg in HISTORY:
            print(msg)