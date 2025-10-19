import os
from ollama import chat

os.environ['OLLAMA_NUM_THREADS'] = '8'
os.environ['OLLAMA_CUDA'] = '1'  # Set to '0' if no GPU available

def run_prompt_full(prompt):
    response = chat(
        model='tinyllama:latest',
        messages=[{'role': 'user', 'content': prompt}],
        stream=False
    )
    # Now response is a dict with keys
    answer = response['message']['content']
    print("Ollama says:", answer)

if __name__ == "__main__":
    run_prompt_full("I am currently locate in Bradford Uinted Kingdom, tell me about that city.")