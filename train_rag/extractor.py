import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample

# --------------------------
# CONFIG
# --------------------------
CORPUS_FILE = "prepared/chunked_docs.jsonl"
FAISS_INDEX_FILE = "faiss_index.idx"
ID_MAP_FILE = "id_map.json"

BI_MODEL_PATH = "biencoder/model"
RERANKER_MODEL_PATH = "reranker/model"

TOP_K = 10  # number of candidates to retrieve

# --------------------------
# LOAD CORPUS
# --------------------------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

corpus = load_jsonl(CORPUS_FILE)
print(f"Loaded {len(corpus)} documents.")

# --------------------------
# LOAD BI-ENCODER AND ENCODE
# --------------------------
bi_encoder = SentenceTransformer(BI_MODEL_PATH)

corpus_embeddings = []
ids = []
batch_size = 64

print("Encoding corpus with bi-encoder...")
for i in tqdm(range(0, len(corpus), batch_size)):
    batch = corpus[i:i+batch_size]
    texts = [d["text"] for d in batch]
    embeddings = bi_encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    corpus_embeddings.append(embeddings)
    ids.extend([d["id"] for d in batch])

corpus_embeddings = np.vstack(corpus_embeddings)
print(f"Corpus encoded: {corpus_embeddings.shape}")

# --------------------------
# BUILD FAISS INDEX
# --------------------------
embedding_dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(corpus_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# Save index
faiss.write_index(index, FAISS_INDEX_FILE)
# Save ID map
with open(ID_MAP_FILE, "w") as f:
    json.dump(ids, f)
print(f"FAISS index saved to {FAISS_INDEX_FILE}")
print(f"ID map saved to {ID_MAP_FILE}")

# --------------------------
# OPTIONAL: QUERY + RERANK
# --------------------------
reranker = CrossEncoder(RERANKER_MODEL_PATH)

query = "What is malnutrition?"
query_emb = bi_encoder.encode([query], convert_to_numpy=True)

# Retrieve top-K candidates from FAISS
distances, indices = index.search(query_emb, TOP_K)
retrieved_ids = [ids[i] for i in indices[0]]
retrieved_texts = [corpus[i]["text"] for i in indices[0]]

# Prepare pairs for reranker
pairs = [[query, text] for text in retrieved_texts]
scores = reranker.predict(pairs)

# Rerank results
reranked = sorted(zip(retrieved_ids, retrieved_texts, scores), key=lambda x: x[2], reverse=True)

print("\nTop results after reranking:")
for doc_id, text, score in reranked:
    print(f"Score: {score:.4f} | ID: {doc_id} | Text snippet: {text[:100]}")
