# 🎬 StudioRAG

> Semantic Q&A over animation & VFX production documents — powered by Claude Haiku + ChromaDB

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built for animation studio knowledge bases. Query technical documentation - rendering papers, pipeline guides, rigging workflows, VFX standards - using natural language.

---

## 📽️ Demo

https://github.com/user-attachments/assets/e13804c1-4511-4aa3-8fe1-7d1bd0b97b29

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  PDF Corpus │────▶│  ingest.py           │────▶│  ChromaDB       │
│  (./docs)   │     │  chunk + embed       │     │  vector store   │
└─────────────┘     │  (MiniLM-L6-v2)      │     └────────┬────────┘
                    └──────────────────────┘              │
                                                          │ cosine similarity (top-6)
                                                          │
┌─────────────┐     ┌──────────────────────┐              │
│  Streamlit  │◀────│  rag.py              │◀─────────────┘
│  UI         │     │  LangChain chain     │
└─────────────┘     │  Claude Haiku (API)  │
                    └──────────────────────┘
```

| Component | Technology |
|---|---|
| LLM | Claude Haiku (Anthropic API) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | ChromaDB |
| RAG orchestration | LangChain 0.3 |
| UI | Streamlit |
| Deployment | Docker Compose |

---

## 📁 Document Corpus

| Category | Documents |
|---|---|
| Rendering / Graphics | RenderMan XPU (CPU+GPU hybrid), Deep Pixel Denoising |
| Pipeline Engineering | Animation Pipeline Automation, Efficient Pipeline Tools |
| Character Animation | Animation Workflow, Metaball Rigging (Pixar Elio) |
| FX / Look Development | Procedural Look Dev (Elio), Shapeshifting FX, Alien Shading |
| Studio Infrastructure | Virtual Production Field Guide (Vol 1 & 2), VFX Naming Standard |

~600 pages of real animation and VFX production documentation.

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (CPU) |
| Chunk size | 800 tokens, 100 overlap |
| Retrieval strategy | Top-6 cosine similarity |
| Retrieval latency (cold start) | ~200ms |
| Retrieval latency (warm avg) | ~35ms |
| LLM | Claude Haiku (Anthropic API) |

---

## 🚀 Quickstart

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 integration enabled
- An [Anthropic API key](https://console.anthropic.com/)
- Your PDF documents in the `./docs` folder

### 1. Clone the repo
```bash
git clone https://github.com/safaa-40/StudioRAG.git
cd StudioRAG
```

### 2. Add your API key
Create a `.env` file in the project root:
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### 3. Add your documents
```bash
cp your_pdfs/*.pdf docs/
```

### 4. Build and start
```bash
docker compose up -d --build
```

### 5. Ingest your documents
```bash
docker exec studiorag_app python ingest.py
```

### 6. Open the UI
Visit [http://localhost:8501](http://localhost:8501)

---

## 💡 Example Queries

- *"What are the stages of an animation pipeline?"*
- *"How does RenderMan XPU handle CPU and GPU rendering?"*
- *"How is motion capture retargeting performed?"*
- *"What is USD and how is it used in production?"*
- *"How are VFX image sequences named across studios?"*
- *"What tools are used for procedural look development?"*
- *"How does denoising work for deep compositing?"*

---

## 📂 Project Structure

```
studiorag/
├── .env                     # Your API key (never committed)
├── .gitignore
├── docker-compose.yml       # Orchestrates the app container
├── docs/                    # Place your PDFs here (not committed)
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── ingest.py            # PDF ingestion and embedding pipeline
    ├── rag.py               # Retrieval + generation logic
    └── ui.py                # Streamlit chat interface
```

---

## 🔒 Security Notes

- `.env` and `docs/` are excluded from version control via `.gitignore`
- The `chroma_db/` vector store is also excluded, regenerate it with `ingest.py`

---

## 📝 Notes

- PDF documents are not included in this repository to keep the repo lightweight, all source documents are publicly available (Pixar SIGGRAPH papers, DreamWorks research, VES guides, and academic theses).
- Cold start retrieval (~200ms) occurs when ChromaDB loads into memory on first query. Subsequent warm queries average ~35ms.
- LLM generation time depends on the Anthropic API response time, typically 2–5 seconds for detailed answers.
