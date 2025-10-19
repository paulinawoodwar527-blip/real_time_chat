from ollama import chat  # Assuming Ollama setup as before
import os
os.environ['OLLAMA_NUM_THREADS'] = '8'
os.environ['OLLAMA_CUDA'] = '1'  # Set to '0' if no GPU available
def make_rag_prompt(retrieved_docs, user_query):
    # Combine retrieved documents into a single string with clear separators
    context_texts = [doc[0] for doc in retrieved_docs]

    # Join the text parts into one context string
    context = "\n\n--- Retrieved Context ---\n\n".join(context_texts)

    # Construct the prompt template
    prompt = f"""
You are an AI assistant. Use the following context to answer the question below persuadably.
I am currently having a interview with my client.

Context:
{context}

Question:
{user_query}

Answer resonably based only on the above context."
"""
    response = chat(
        model='tinyllama:latest',  # Your Ollama model name
        messages=[{'role': 'user', 'content': prompt}]
    )
    answer = response['message']['content']
    print("\nOllama-generated Answer:\n", answer)
    return answer
    # return prompt

# Example usage
# retrieved_texts = [
#     "Computer vision is a field that enables computers to interpret and process visual data.",
#     "It involves techniques like object detection, segmentation, and image classification."
# ]

# user_q = "What is computer vision?"

# prompt_text = make_rag_prompt(retrieved_texts, user_q)
# print(prompt_text)