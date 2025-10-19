import os
import subprocess
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

from pdf_extractor import pdf_extractor
import json

# Text splitting into chunks (e.g., 300 characters)
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = []
chunks.extend(splitter.split_text(pdf_extractor("C:\\Document3.pdf")))

# Load embedding model via Sentence-Transformers compatible with Ollama workflow
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# Normalize embeddings for FAISS
faiss.normalize_L2(embeddings)

# Create FAISS index (FlatL2 for cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

filename = "my_faiss_index.faiss"
faiss.write_index(index, filename)

print(f"FAISS index saved to {filename}")

metadata = {i: chunk for i, chunk in enumerate(chunks)}

output_filename = "metadata.json"
with open(output_filename, 'w') as json_file:
    json.dump(metadata, json_file, indent=4)
# # 5. (Optional) Load the saved index to verify


# Metadata for mapping FAISS results back to text chunks
# metadata = {i: chunk for i, chunk in enumerate(chunks)}

# def query_faiss_index(query, index, embedding_model, metadata, top_k=3):
#     query_embed = embedding_model.encode([query])
#     query_embed = np.array(query_embed).astype("float32")
#     faiss.normalize_L2(query_embed)
#     D, I = index.search(query_embed, top_k)
#     results = [(metadata[idx], D[0][i]) for i, idx in enumerate(I[0])]
#     return results

# def query_ollama(model, prompt):
#     try:
#         result = subprocess.run(
#             ["ollama", "run", model],
#             input=prompt,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             encoding="utf-8",
#             check=True
#         )
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         return f"Error querying Ollama: {e.stderr.strip()}"

# # Example query
# user_query = "Explain the content of document 1"

# # Retrieve from FAISS
# faiss_results = query_faiss_index(user_query, index, embedding_model, metadata, top_k=3)
# retrieved_text = "\n\n".join([text for text, score in faiss_results])

# # Prompt for Ollama with context
# prompt = f"Answer the question based on the following context:\n\nContext: {retrieved_text}\n\nQuestion: {user_query}"

# # Query Ollama for answer generation (replace llama3.2:latest with your model)
# answer = query_ollama("llama3.2:latest", prompt)

# print("Answer:", answer)
