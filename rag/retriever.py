import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from answer_generator import make_rag_prompt

# Load documents or metadata from JSON
with open("metadata.json", "r") as file:
    metadata = json.load(file)
texts = list(metadata.values())

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode texts to embeddings
embeddings = model.encode(texts, convert_to_numpy=True)
embeddings = embeddings.astype("float32")

# Normalize embeddings for cosine similarity search
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
num_points = embeddings.shape[0]
nlist = min(100, num_points)  # number of clusters
m = 8        # number of subquantizers for PQ
nbits = 8    # bits per subvector

if num_points < 50:
    print("Using flat index due to small data size.")
    index = faiss.IndexFlatIP(dimension)
else:
    nlist = min(256, num_points)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
    index.train(embeddings)

index.add(embeddings)

# Save the index for reuse (optional)
faiss.write_index(index, "faiss_ivfpq.index")

def search(query, top_k=5):
    # Encode and normalize query
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Set number of visited clusters for search accuracy/speed tradeoff
    index.nprobe = 10

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        key = str(idx)
        results.append((metadata.get(key, ""), float(dist)))
    return results

if __name__ == "__main__":
    # Example query
    query_text = "what is most challengin AI project you've built?"
    results = search(query_text)
    for text, score in results:
        print(f"Score: {score:.4f}, Text: {text[:200]}...")

    print(make_rag_prompt(results, query_text))