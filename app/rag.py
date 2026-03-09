"""
rag.py: Retrieval-Augmented Generation core.
Handles vector retrieval from ChromaDB and answer generation via Claude (Anthropic).
"""

import os
import time
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"  # Fast and cheap, perfect for demos
TOP_K = 6

PROMPT_TEMPLATE = """You are StudioRAG, a senior technical expert in animation and VFX production pipelines.

Your job is to give detailed, thorough, well-structured answers to questions from artists and pipeline engineers.

Rules:
- Use ONLY the context provided below to answer. Do not make things up.
- Be as detailed and complete as possible. Do not truncate your answer.
- Structure your answer with clear sections when appropriate.
- Include specific technical details, names, tools, and processes mentioned in the context.
- If the context is insufficient to fully answer, say what you can and note what is missing.

Context:
{context}

Question: {question}

Detailed Answer:"""


def get_vectorstore() -> Chroma:
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="studiorag",
    )


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.3,
        max_tokens=2048,
    )


def build_rag_chain(retriever, llm):
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def retrieve_sources(vectorstore: Chroma, query: str) -> tuple[list[dict], float]:
    """Return the top-k source chunks and the retrieval latency in ms."""
    t0 = time.perf_counter()
    docs = vectorstore.similarity_search(query, k=TOP_K)
    latency_ms = (time.perf_counter() - t0) * 1000

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:400].strip(),
        }
        for doc in docs
    ]
    return sources, latency_ms