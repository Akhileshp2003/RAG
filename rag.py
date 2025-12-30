import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ===== Local Embeddings (Sentence Transformers) =====
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===== Local LLM (LLaMA via HuggingFace / llama-cpp) =====
from langchain_community.llms import LlamaCpp


# =========================
# LOCAL LLM CONFIG
# =========================

# Download a GGUF model first (example):
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Place it locally and update the path below

LLAMA_MODEL_PATH = r"C:\Users\DELL\Desktop\shellkode\nandha_college_demo\models\llama-2-7b-chat.Q2_K.gguf"


llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    temperature=0.0,
    max_tokens=512,
    n_ctx=4096,
    verbose=False,
)


def call_llm(prompt: str) -> str:
    """Call local LLaMA model"""
    t0 = time.time()
    response = llm.invoke(prompt)
    print(f"[INFO] LLM call took {time.time() - t0:.2f}s")
    return response


# =========================
# BUILD VECTOR STORE
# =========================
def build_vector_store(pdf_path: str) -> FAISS:
    """
    Load PDF → split → embed (Sentence Transformers) → FAISS
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# =========================
# ASK QUESTION (RAG)
# =========================
def ask_pdf_question(
    vector_store: FAISS,
    question: str,
    k: int = 3
) -> str:
    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join(
        f"Page {doc.metadata.get('page')}:\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""
You are a document-based assistant.

Answer the question strictly using the context below.
If the answer is not present, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""

    return call_llm(prompt)


# =========================
# MAIN (CLI DEMO)
# =========================
if __name__ == "__main__":
    PDF_PATH = "resume.pdf"

    print("Building FAISS index with Sentence Transformers...")
    vector_store = build_vector_store(PDF_PATH)

    print("Local RAG system ready. Ask questions!\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_pdf_question(vector_store, query)
        print("\nAnswer:\n", answer)
        print("-" * 60)