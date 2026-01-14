import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


embed_model = SentenceTransformer("all-MiniLM-L6-v2")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=200
)


with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = text.split(". ")


embeddings = embed_model.encode(chunks)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

st.title("📄 Free RAG System (Offline)")

query = st.text_input("Ask a question")

if query:
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=2)

    context = " ".join([chunks[i] for i in I[0]])

    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

    result = generator(prompt)
    st.subheader("Answer")
    st.write(result[0]["generated_text"])
