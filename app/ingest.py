"""
ingest.py: Load PDFs from /docs, chunk, embed, and store in ChromaDB.
Run this once before launching the UI:
    docker exec studiorag_app python ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DOCS_PATH = os.getenv("DOCS_PATH", "./docs")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_pdfs(docs_path: str) -> list:
    pdf_files = list(Path(docs_path).glob("**/*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {docs_path}")
        sys.exit(1)

    print(f"[INFO] Found {len(pdf_files)} PDF(s):")
    all_docs = []
    for pdf in pdf_files:
        print(f"  • {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        # Attach source filename to metadata
        for doc in docs:
            doc.metadata["source"] = pdf.name
        all_docs.extend(docs)

    print(f"[INFO] Loaded {len(all_docs)} pages total.")
    return all_docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


def embed_and_store(chunks: list):
    print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    print(f"[INFO] Embedding and storing in ChromaDB at {CHROMA_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="studiorag",
    )
    print(f"[INFO] Done. {vectorstore._collection.count()} chunks stored.")
    return vectorstore


if __name__ == "__main__":
    print("=" * 50)
    print("StudioRAG — Ingestion Pipeline")
    print("=" * 50)
    docs = load_pdfs(DOCS_PATH)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)
    print("[SUCCESS] Ingestion complete. You can now launch the UI.")
