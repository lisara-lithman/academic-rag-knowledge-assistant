# 🎓 Academic RAG Knowledge Assistant (OSSA)

An advanced, interactive Artificial Intelligence Tutor built to assist university students with the **Operating Systems & System Administration (OSSA)** module. 

Traditional studying often requires manually searching through hundreds of PDF lecture slides, tutorial sheets, and past papers to find specific definitions or concepts. This project solves that problem by utilizing an advanced **Retrieval-Augmented Generation (RAG)** pipeline. When a student asks a question, the system searches the actual course materials, extracts the most relevant slides, and generates a precise, hallucination-free answer—complete with exact file and page citations.

---

## 🌟 Deep Dive: Architecture & Engineering Features

This is not a basic RAG implementation. It includes advanced optimizations typically found in production-grade systems to maximize retrieval recall and generation accuracy.

### 1. Semantic Document Chunking & Ingestion
- **Smart Parsing:** Raw PDF lecture slides, tutorials, and past papers are parsed using `PyPDF2`.
- **Semantic Chunking:** Instead of arbitrarily slicing text every 1,000 characters (which breaks sentences in half), the `ingest.py` script uses regex-based semantic chunking. It intelligently splits text by paragraphs and sentences, grouping them together up to 800 characters to keep semantic concepts intact.
- **Embedding:** Text chunks are vectorized using OpenAI's powerful `text-embedding-3-large` model and stored in a local **ChromaDB** vector database.

### 2. Query Routing & Domain Rejection
- When a user asks a question, an LLM router intercepts it and assigns one of three actions: `SEARCH`, `REFINE`, or `CHAT`.
- **Strict Domain Rejection:** The router is strictly anchored to the Operating Systems domain. If a student attempts to ask about out-of-scope topics (e.g., Cooking, Node.js, Web Development), the router politely refuses to answer, keeping the assistant strictly focused on academic material.

### 3. Context-Aware Query Expansion
- **Handling Ambiguity:** User queries are often vague (e.g., "What are the advantages of that?"). The system uses the LLM and the recent conversation history to deduce the context.
- **Multi-Query Generation:** To maximize vector recall, the system rewrites the user's intent into **two** optimized search queries (one broad, one specific) and searches the vector database with both, merging and deduplicating the results.

### 4. Two-Stage Retrieval Pipeline
- **Stage 1 (Dense Retrieval):** The system searches ChromaDB with the expanded queries to quickly fetch the top 8 most semantically similar slide chunks.
- **Stage 2 (Cross-Encoder Reranking):** Dense retrieval can sometimes return loosely related chunks. To ensure precision, a heavier `ms-marco-MiniLM-L-6-v2` cross-encoder model meticulously scores and re-orders the retrieved chunks, filtering out anything that doesn't pass a strict relevance threshold before handing it to the generator.

### 5. Hybrid LLM Generation Engine
- **Powered by GPT-4o:** The final answer generation is driven by OpenAI's `gpt-4o` model for superior reasoning.
- **Hybrid Fallback Prompting:** The LLM is strictly instructed to use the provided lecture slides as its primary source of truth. However, if the reranker determines that the slides lack the necessary information for a highly advanced OS topic (e.g., TLBs), the LLM gracefully falls back to its own expert knowledge, explicitly warning the student that the information is supplementary.

### 6. Robust Rate Limit Handling
- **Exponential Backoff:** Because the architecture makes multiple LLM calls per query (Routing + Generation), free-tier API keys can quickly hit rate limits. The `retrieval.py` script features a custom, robust retry loop that catches HTTP 429 and 503 timeouts, sleeping and retrying exponentially in the background up to 5 times (absorbing up to a minute of rate limits seamlessly).
- **Pre-warmed UI:** To prevent Gradio from throwing timeout errors on the very first cold-start query, the database and LLM clients are pre-loaded into RAM during the UI startup sequence.

### 7. Citation-Driven User Interface
- A custom-designed Gradio web UI that features a split-screen design. 
- The left side handles the dynamic, token-by-token streaming of the chat answer.
- The right side dynamically displays the source materials, relevance scores, and precise page numbers used to answer the current question.

---

## 📊 RAGAS Evaluation Framework

This system was rigorously tested against a stress-test dataset of noisy, out-of-scope, and ambiguous queries. We utilized the **RAGAS** (Retrieval Augmented Generation Assessment) framework to measure Faithfulness, Answer Relevance, and Context Recall. 

The system comfortably crossed the production-ready 0.80 RAGAS threshold. 

**[👉 Click here to read the full Evaluation Report (EVALUATION.md)](./EVALUATION.md)**

---

## 🏗️ Folder Structure

```text
academic-rag-knowledge-assistant/
├── ingest.py           # Pipeline for PDF parsing -> Semantic Chunking -> ChromaDB
├── retrieval.py        # The core RAG logic (Routing, Expanding, Reranking)
├── ui.py               # The Gradio web interface & streaming management
├── EVALUATION.md       # Deep-dive report on RAG metrics and engineering loop
├── requirements.txt    # Python dependencies
├── .env                # API Keys (Not tracked in Git)
├── vector_db/          # The local ChromaDB vector database (Auto-generated)
└── knowledge_base/     # Place your course PDFs here
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher
- Get a free OpenAI API key: [platform.openai.com](https://platform.openai.com/)

### 2. Clone and Configure
```bash
git clone https://github.com/lisara-lithman/academic-rag-knowledge-assistant.git
cd academic-rag-knowledge-assistant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file in the root directory:
```env
# Add your API key here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Build the Vector Database
Add your course PDFs to the `knowledge_base/operating_systems/lecture/` folder. Then run:
```bash
python ingest.py
```
*Note: This process takes a few minutes on the first run as it downloads the HuggingFace reranker model and generates embeddings.*

### 5. Launch the Assistant
```bash
python ui.py
```
Open the provided local URL (usually `http://127.0.0.1:7860`) in your web browser.

---

## ⚠️ Troubleshooting & Known Issues

- **OpenAI Rate Limits:** If using the free tier of OpenAI, you may hit strict Requests-Per-Minute (RPM) limits if you ask complex questions back-to-back. The backend script will gracefully absorb these rate limits by sleeping in the background. If the web UI times out or throws an error, simply wait 5-10 minutes for your OpenAI quota to cool down, refresh the page, and try again.
- **Port Conflicts:** If `python ui.py` throws an `OSError: Cannot find empty port`, it means the server is already running in the background. Kill it using `kill -9 $(lsof -t -i:7860)` on Mac/Linux.

---

## 📜 License
This project is for educational use and development. Built to streamline university revision and demonstrate advanced, production-grade implementations of localized Retrieval-Augmented Generation.
