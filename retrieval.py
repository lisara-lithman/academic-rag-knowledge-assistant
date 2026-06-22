import os
import time
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from google import genai
from groq import Groq
from sentence_transformers import CrossEncoder

# Load configurations
load_dotenv()

DB_DIR = os.getenv("PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = "operating_systems"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global model caches
_openai_ef = None
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


def get_openai_ef():
    """Get the OpenAI Embedding Function for ChromaDB (lazy-loaded)."""
    global _openai_ef
    if _openai_ef is None:
        print("Loading OpenAI embedding function...")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in .env!")
        _openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name=EMBEDDING_MODEL_NAME
        )
    return _openai_ef


def load_rerank_model():
    """Load the local Cross-Encoder model for reranking (lazy-loaded)."""
    global _rerank_model
    if _rerank_model is None:
        print("Loading local Cross-Encoder reranker...")
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


def _is_rate_limit_error(e):
    err_str = str(e).lower()
    return "429" in err_str or "rate limit" in err_str or "quota" in err_str


def call_with_retry(func, max_retries=3, initial_backoff=2):
    """Executes a function with exponential backoff if a rate limit error is encountered."""
    retries = 0
    backoff = initial_backoff
    while True:
        try:
            return func()
        except Exception as e:
            if _is_rate_limit_error(e) and retries < max_retries:
                print(f"Rate limit hit. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            else:
                raise e


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
        "- Output 'SEARCH' if the student asks a new question OR introduces a new technical term/concept as a follow-up.\n"
        "- Output 'CHAT' only for greetings or purely conversational filler.\n\n"
        "REWRITING RULES (If SEARCH):\n"
        "- Expand messy, informal inputs into clean, formal search queries.\n"
        "- If the student's question is vague or broad (e.g., 'Explain memory management'), break it down into TWO specific sub-queries covering different aspects of the topic (e.g., allocation vs paging).\n"
        "- If the question is already highly specific, just output one rewritten query.\n"
        "- Preserve all core concepts from the student's input.\n"
        "- DO NOT hallucinate unrelated domains like databases or web development. Keep the focus strictly on Operating Systems.\n\n"
        "Respond EXACTLY in this format:\n"
        "DECISION: [SEARCH | REFINE | CHAT]\n"
        "QUERY1: [Your first rewritten query, or the topic if REFINE]\n"
        "QUERY2: [Your second rewritten query if the topic is broad/ambiguous, otherwise omit this line]\n"
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
        rewritten_queries = []
        chat_response = ""

        for line in response_text.split("\n"):
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("QUERY1:"):
                q1 = line.replace("QUERY1:", "").strip()
                if q1 and q1 != "[Your first rewritten query, or the topic if REFINE]":
                    rewritten_queries.append(q1)
            elif line.startswith("QUERY2:"):
                q2 = line.replace("QUERY2:", "").strip()
                if q2 and q2 != "[Your second rewritten query if the topic is broad/ambiguous, otherwise omit this line]":
                    rewritten_queries.append(q2)
            elif line.startswith("RESPONSE:"):
                chat_response = line.replace("RESPONSE:", "").strip()

        # Clean quotes if model wrapped them
        rewritten_queries = [q[1:-1] if q.startswith('"') and q.endswith('"') else q for q in rewritten_queries]

        if not rewritten_queries:
            rewritten_queries = [query]

        # Fallback safety
        if decision not in ["SEARCH", "REFINE", "CHAT"]:
            decision = "SEARCH"

        # If there's no history, we cannot refine a previous answer
        if not history:
            decision = "SEARCH" if decision == "REFINE" else decision

        # Fallback chat response if the LLM did not fill the RESPONSE field
        if decision == "CHAT" and not chat_response:
            chat_response = "Hello! 👋 Feel free to ask me anything about Operating Systems & System Administration!"

        print(f"User Input: '{query}' -> Decision: {decision} | Rewritten: {rewritten_queries}")
        return decision, rewritten_queries, chat_response

    except Exception as e:
        print(f"Warning: Query rewrite failed ({e}). Defaulting to SEARCH.")
        return "SEARCH", [query], ""


def retrieve_chunks(query, n_results=5):
    """Embed the query and retrieve top-K matching chunks from ChromaDB."""
    openai_ef = get_openai_ef()

    # 1. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    # 2. Query ChromaDB directly with text (Chroma handles embedding via openai_ef)
    results = collection.query(
        query_texts=[query],
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
    Returns: (decision, rewritten_queries, final_context, chat_response)
    chat_response is only populated when decision == 'CHAT'.
    """
    # 1. Rewrite user's query and get decision
    decision, rewritten_queries, chat_response = rewrite_query(query, history=history)

    if decision != "SEARCH":
        # No new search needed; return empty chunks and pass chat_response through
        return decision, rewritten_queries, [], chat_response

    # 2. Retrieve using original query
    all_chunks = retrieve_chunks(query, n_results=5)

    # 3. Retrieve using all rewritten queries
    for sub_query in rewritten_queries:
        sub_chunks = retrieve_chunks(sub_query, n_results=5)
        all_chunks = merge_and_deduplicate(all_chunks, sub_chunks)

    # 5. Rerank against the ORIGINAL user query
    # If it generated multiple queries (ambiguous), boost top_k
    if len(rewritten_queries) > 1:
        top_k = 8

    final_context = rerank_chunks(query, all_chunks, top_k=top_k)

    return decision, rewritten_queries, final_context, ""


if __name__ == "__main__":
    # Test execution
    test_query = "What are the states of a process?"
    print(f"Testing search pipeline with query: '{test_query}'\n")
    decision, rewritten, results = search_pipeline(test_query)
