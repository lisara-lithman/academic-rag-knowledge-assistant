import os
import chromadb
from dotenv import load_dotenv
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load configurations
load_dotenv()

DB_DIR = os.getenv("PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = "operating_systems"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global model caches
_embedding_model = None
_rerank_model = None
_llm_client = None
_llm_provider = None


def get_llm_client():
    """Initialize either Groq or Gemini client based on environment variables."""
    global _llm_client, _llm_provider
    if _llm_client is not None:
        return _llm_client, _llm_provider

    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if groq_key and not groq_key.startswith("your_"):
        print("Using Groq API for LLM reasoning...")
        # Groq client automatically picks up GROQ_API_KEY from environment
        _llm_client = Groq()
        _llm_provider = "groq"
    elif gemini_key and not gemini_key.startswith("your_"):
        print("Using Gemini API for LLM reasoning...")
        # Gemini client automatically picks up GEMINI_API_KEY from environment
        _llm_client = genai.Client()
        _llm_provider = "gemini"
    else:
        raise ValueError(
            "No valid API keys found in .env! "
            "Please configure either GROQ_API_KEY or GEMINI_API_KEY."
        )

    return _llm_client, _llm_provider


def load_embedding_model():
    """Load the local sentence-transformer embedding model (lazy-loaded)."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def load_rerank_model():
    """Load the local Cross-Encoder model for reranking (lazy-loaded)."""
    global _rerank_model
    if _rerank_model is None:
        print("Loading local Cross-Encoder reranker...")
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


def rewrite_query(query, history=None):
    """
    Analyzes student query with context of history.
    Returns: (decision, rewritten_query)
    decision: "SEARCH", "REFINE", or "CHAT"
    """
    client, provider = get_llm_client()

    history_context = ""
    if history:
        history_context = "Conversation History:\n"
        for user_msg, bot_msg in history[-3:]:
            if bot_msg:
                history_context += f"Student: {user_msg}\nTutor: {bot_msg[:150]}...\n"
            else:
                history_context += f"Student: {user_msg}\n"
        history_context += "\n"

    prompt = (
        "You are an AI assistant for a university module search engine. Your task is to analyze the student's latest input "
        "and determine if they are asking a new question that requires searching the database for new course materials, "
        "or if they are instructing you to refine, summarize, translate, shorten, or visualize the current topic/previous answer.\n\n"
        "Respond in the following format:\n"
        "DECISION: [SEARCH | REFINE | CHAT]\n"
        "QUERY: [If SEARCH, write a descriptive keyword-rich search query. If REFINE or CHAT, write the key topic of discussion.]\n\n"
        f"{history_context}"
        f"Latest Student Input: \"{query}\""
    )

    try:
        if provider == "groq":
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response_text = completion.choices[0].message.content.strip()
        elif provider == "gemini":
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            response_text = response.text.strip()

        # Parse decision and query
        decision = "SEARCH"
        rewritten = query

        for line in response_text.split("\n"):
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("QUERY:"):
                rewritten = line.replace("QUERY:", "").strip()

        # Clean quotes if model wrapped them
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]

        # Fallback safety
        if decision not in ["SEARCH", "REFINE", "CHAT"]:
            decision = "SEARCH"

        print(f"User Input: '{query}' -> Decision: {decision} | Rewritten: '{rewritten}'")
        return decision, rewritten

    except Exception as e:
        print(f"Warning: Query rewrite failed ({e}). Defaulting to SEARCH.")
        return "SEARCH", query


def retrieve_chunks(query, n_results=5):
    """Embed the query and retrieve top-K matching chunks from ChromaDB."""
    embedding_model = load_embedding_model()

    # 1. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # 2. Embed the query vector
    query_vector = embedding_model.encode(query).tolist()

    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['documents', 'metadatas']
    )

    # Format the results into clean dictionaries
    formatted_chunks = []
    if results and results['documents']:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]

        for idx, (doc, meta, chunk_id) in enumerate(zip(documents, metadatas, ids)):
            formatted_chunks.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta
            })

    return formatted_chunks


def merge_and_deduplicate(chunks_a, chunks_b):
    """Merge two lists of retrieved chunks and remove duplicates by chunk ID."""
    seen_ids = set()
    unique_chunks = []
    for chunk in chunks_a + chunks_b:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            unique_chunks.append(chunk)
    return unique_chunks


def rerank_chunks(query, chunks, top_k=4):
    """
    Rerank a set of unique chunks against the original user query
    using a local Cross-Encoder model.
    """
    if not chunks:
        return []

    reranker = load_rerank_model()

    # Format pairs of [query, chunk_text] for scoring
    pairs = [[query, chunk["text"]] for chunk in chunks]
    print(f"Reranking {len(chunks)} unique chunks...")
    scores = reranker.predict(pairs)

    # Add score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by score in descending order
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Keep only the top K chunks
    top_chunks = chunks[:top_k]

    print("\n--- Top Reranked Contexts ---")
    for i, chunk in enumerate(top_chunks):
        src = chunk['metadata']['source']
        page = chunk['metadata']['page']
        score = chunk['rerank_score']
        print(f"[{i+1}] Score: {score:.4f} | Source: {src} (Page {page})")
    print("-----------------------------\n")

    return top_chunks


def search_pipeline(query, history=None, top_k=4):
    """
    The full advanced retrieval pipeline:
    User Query -> Query Rewrite Decision -> Dual Search -> Merge & Deduplicate -> Rerank -> Top Chunks
    Returns: (decision, rewritten_query, final_context)
    """
    # 1. Rewrite user's query and get decision
    decision, rewritten_query = rewrite_query(query, history=history)

    if decision != "SEARCH":
        # No new search needed, return empty chunks
        return decision, rewritten_query, []

    # 2. Retrieve using original query
    chunks_original = retrieve_chunks(query, n_results=5)

    # 3. Retrieve using rewritten query
    chunks_rewritten = retrieve_chunks(rewritten_query, n_results=5)

    # 4. Merge and deduplicate
    merged_chunks = merge_and_deduplicate(chunks_original, chunks_rewritten)

    # 5. Rerank against the ORIGINAL user query
    final_context = rerank_chunks(query, merged_chunks, top_k=top_k)

    return decision, rewritten_query, final_context


if __name__ == "__main__":
    # Test execution
    test_query = "What are the states of a process?"
    print(f"Testing search pipeline with query: '{test_query}'\n")
    decision, rewritten, results = search_pipeline(test_query)
