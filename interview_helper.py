
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
from ollama import chat  # Assuming Ollama setup as before
import os

os.environ['OLLAMA_NUM_THREADS'] = '8'
os.environ['OLLAMA_CUDA'] = '1'  # Set to '0' if no GPU available
# Speech-to-text capturing
def listen_and_transcribe():
    r = sr.Recognizer()
    # with sr.Microphone() as source:
    #     print("Adjusting for ambient noise...")
    #     r.adjust_for_ambient_noise(source, duration=1)
    #     print("Please speak your question:")
    #     audio = r.listen(source)
    #     try:
    #         text = r.recognize_google(audio)
    #         print("You said:", text)
    #         return text
    #     except sr.UnknownValueError:
    #         print("Speech not understood, please try again.")
    #         return ""
    #     except sr.RequestError as e:
    #         print("Could not request results; {0}".format(e))
    #         return ""
    AUDIO_FILE = "C:\\Users\\WWW\\Downloads\\Speech-Enhancement-Spectral-Subtraction-master\\Speech_signal_files\\speech_ref.wav"  # Replace with your audio file path

    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.listen(source)  # Read the entire audio file

    try:
        text = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Resume chunks and embeddings
resume_chunks = [
    "Experienced AI engineer with expertise in machine learning and data science.",
    "Proficient in Python, JavaScript, and cloud services.",
    "Built production-level AI systems for reputation management and pricing models."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(resume_chunks)

# Initialize hnswlib index
dim = embeddings.shape[1]
num_elements = len(resume_chunks)
index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=200, M=16)
index.add_items(embeddings)
index.set_ef(50)  # ef should always be > k

def search_resume(question, top_k=2):
    q_emb = model.encode([question])
    labels, distances = index.knn_query(q_emb, k=top_k)
    results = [resume_chunks[label] for label in labels[0]]
    return "\n".join(results)

# Ollama LLM query
def get_ai_answer(question, context):
    prompt = (f"Use the resume information below to answer this interview question:\n\n{context}\n\n"
              f"Question: {question}\nAnswer:")
    response = chat(
        model='tinyllama:latest',  # Your Ollama model name
        messages=[{'role': 'user', 'content': prompt}]
    )
    answer = response['message']['content']
    print("\nOllama-generated Answer:\n", answer)
    return answer

# Main
if __name__ == "__main__":
    question_text = listen_and_transcribe()
    if question_text:
        resume_context = search_resume(question_text)
        get_ai_answer(question_text, resume_context)
    else:
        print("No valid question detected.")