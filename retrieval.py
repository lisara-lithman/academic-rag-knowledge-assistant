import os
import time
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


def _is_rate_limit_error(e):
    """Return True if the exception is an HTTP 429 rate-limit error from any LLM provider."""
    err = str(e).lower()
    return "429" in err or "rate limit" in err or "too many requests" in err or "ratelimit" in err


def call_with_retry(fn, max_retries=3, base_wait=5):
    """
    Call fn() and retry on rate-limit errors using exponential backoff.
    Wait times: 5s → 10s → 20s between successive retries.
    Raises the original exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)   # 5, 10, 20 seconds
                print(f"⚠️  Rate limit hit. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise


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
        "You are the query router and rewriter for a university 'Operating Systems and System Administration' (OSSA) module.\n"
        "Your task is to analyze the student's latest input in the context of the conversation history.\n\n"
        "ROUTING RULES:\n"
        "- Output 'REFINE' ONLY if the student asks to format, summarize, shorten, or clarify the EXACT SAME concept from the previous turn.\n"
        "- Output 'SEARCH' if the student asks a new question OR introduces a new technical term/concept as a follow-up (e.g., 'What about semaphores?', 'Explain threads vs processes').\n"
        "- Output 'CHAT' only for greetings or purely conversational filler.\n\n"
        "REWRITING RULES (If SEARCH):\n"
        "- Expand messy, informal inputs into clean, formal search queries.\n"
        "- Preserve all core concepts from the student's input (e.g. 'conditions').\n"
        "- DO NOT hallucinate unrelated domains like databases or web development. Keep the focus strictly on Operating Systems.\n\n"
        "Respond EXACTLY in this format:\n"
        "DECISION: [SEARCH | REFINE | CHAT]\n"
        "QUERY: [Your rewritten query, or the topic if REFINE]\n"
        "RESPONSE: [Only fill this if DECISION is CHAT — write a short, warm, natural conversational reply to the student]\n\n"
        f"{history_context}"
        f"Latest Student Input: \"{query}\""
    )

    try:
        def _call_llm():
            if provider == "groq":
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return completion.choices[0].message.content.strip()
            elif provider == "gemini":
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                return response.text.strip()

        response_text = call_with_retry(_call_llm)

        # Parse decision, query and optional chat response
        decision = "SEARCH"
        rewritten = query
        chat_response = ""

        for line in response_text.split("\n"):
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("QUERY:"):
                rewritten = line.replace("QUERY:", "").strip()
            elif line.startswith("RESPONSE:"):
                chat_response = line.replace("RESPONSE:", "").strip()

        # Clean quotes if model wrapped them
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]

        # Fallback safety
        if decision not in ["SEARCH", "REFINE", "CHAT"]:
            decision = "SEARCH"

        # If there's no history, we cannot refine a previous answer
        if not history:
            decision = "SEARCH" if decision == "REFINE" else decision

        # Fallback chat response if the LLM did not fill the RESPONSE field
        if decision == "CHAT" and not chat_response:
            chat_response = "Hello! 👋 Feel free to ask me anything about Operating Systems & System Administration!"

        print(f"User Input: '{query}' -> Decision: {decision} | Rewritten: '{rewritten}'")
        return decision, rewritten, chat_response

    except Exception as e:
        print(f"Warning: Query rewrite failed ({e}). Defaulting to SEARCH.")
        return "SEARCH", query, ""


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


RELEVANCE_THRESHOLD = 0.0

def rerank_chunks(query, chunks, top_k=4):
    """
    Rerank a set of unique chunks against the original user query
    using a local Cross-Encoder model.
    Chunks scoring below RELEVANCE_THRESHOLD are discarded as irrelevant.
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

    # Filter out chunks that fall below the relevance threshold
    relevant_chunks = [c for c in chunks if c["rerank_score"] >= RELEVANCE_THRESHOLD]
    filtered_count = len(chunks) - len(relevant_chunks)
    if filtered_count > 0:
        print(f"Threshold filter: removed {filtered_count} chunk(s) with score < {RELEVANCE_THRESHOLD}")

    # Keep only the top K from the relevant chunks
    top_chunks = relevant_chunks[:top_k]

    print("\n--- Top Reranked Contexts ---")
    if top_chunks:
        for i, chunk in enumerate(top_chunks):
            src = chunk['metadata']['source']
            page = chunk['metadata']['page']
            score = chunk['rerank_score']
            print(f"[{i+1}] Score: {score:.4f} | Source: {src} (Page {page})")
    else:
        print("No chunks passed the relevance threshold — query is likely out of scope.")
    print("-----------------------------\n")

    return top_chunks


def search_pipeline(query, history=None, top_k=6):
    """
    The full advanced retrieval pipeline:
    User Query -> Query Rewrite Decision -> Dual Search -> Merge & Deduplicate -> Rerank -> Top Chunks
    Returns: (decision, rewritten_query, final_context, chat_response)
    chat_response is only populated when decision == 'CHAT'.
    """
    # 1. Rewrite user's query and get decision
    decision, rewritten_query, chat_response = rewrite_query(query, history=history)

    if decision != "SEARCH":
        # No new search needed; return empty chunks and pass chat_response through
        return decision, rewritten_query, [], chat_response

    # 2. Retrieve using original query
    chunks_original = retrieve_chunks(query, n_results=5)

    # 3. Retrieve using rewritten query
    chunks_rewritten = retrieve_chunks(rewritten_query, n_results=5)

    # 4. Merge and deduplicate
    merged_chunks = merge_and_deduplicate(chunks_original, chunks_rewritten)

    # 5. Rerank against the ORIGINAL user query
    final_context = rerank_chunks(query, merged_chunks, top_k=top_k)

    return decision, rewritten_query, final_context, ""


if __name__ == "__main__":
    # Test execution
    test_query = "What are the states of a process?"
    print(f"Testing search pipeline with query: '{test_query}'\n")
    decision, rewritten, results = search_pipeline(test_query)
