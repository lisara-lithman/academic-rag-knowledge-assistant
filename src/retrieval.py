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

# --- GLOBAL PROMPTS ---
ROUTER_PROMPT = """You are a router for a university 'Operating Systems and System Administration' (OSSA) module.
Your job is to classify the student's input into exactly ONE category. Students are real university students — they may write formally, informally, with slang, abbreviations, typos, or incomplete sentences.

Categories:

- SEARCH: The student is asking about an OS or System Administration concept — regardless of how they phrase it.
  This includes formal questions, informal questions, slang, abbreviations, and partial sentences.
  OS/SA topics include: processes, threads, CPU scheduling, memory management, virtual memory, paging, segmentation,
  deadlocks, semaphores, mutexes, file systems, I/O, system calls, kernel, interrupts, context switching,
  virtualisation, containers, Linux commands, shell scripting, permissions, users/groups, services, networking basics.
  Examples of SEARCH (all of these are SEARCH):
    - "explain deadlocks"
    - "wtf is a semaphore lol"
    - "pls explain pcb simply"
    - "diff between process and thread"
    - "how does paging work"
    - "what happens during a context switch"
    - "i dont get virtual memory can u explain"
    - "scheduling algorithms"
    - "what is a zombie process"

- REFINE: The student is NOT asking a new technical question. They want the PREVIOUS answer changed in format, length, depth, or style.
  This applies when the message is a formatting or follow-up instruction that only makes sense given a prior answer.
  Examples of REFINE (all of these are REFINE when there is history):
    - "can you give more details"
    - "make it shorter"
    - "explain that again but simpler"
    - "give me one sentence"
    - "summarise it"
    - "give an exam-style answer"
    - "bullet points please"
    - "give me a definition only"
    - "tldr"
    - "give me an example"
    - "i still dont understand, try again"
    - "what do you mean by [term from previous answer]"
    - "elaborate on [point from previous answer]"
    - "can you explain it differently"
    - "in simple terms"
    - "how would i write this in an exam"
  CRITICAL RULE: If there is conversation history AND the student's message does NOT introduce a brand new OS/SA topic, classify as REFINE — even if the message contains no OS keywords.

- CHAT: Pure conversational filler with no question or request.
  Examples: "hello", "hi", "thanks", "ok", "great", "got it", "cool", "cheers", "you're welcome".

- OUT_OF_SCOPE: The student is asking about something completely unrelated to OS/SA AND it is clearly not a follow-up to a previous answer.
  Examples of OUT_OF_SCOPE topics:
    - Web frameworks: Flask, Django, FastAPI, Express, Spring
    - Frontend: React, Angular, Vue, HTML, CSS, JavaScript
    - Databases: SQL, MySQL, PostgreSQL, MongoDB, NoSQL
    - AI/ML: machine learning, neural networks, TensorFlow, ChatGPT
    - Other subjects: mathematics, physics, history, geography, biology
    - General programming not related to OS: "how do I sort a list in Python"
    - Personal/meta questions with no history: "who are you", "what can you do"

DECISION PRIORITY (apply in order):
1. If the message is pure filler (hello, thanks) → CHAT
2. If there IS conversation history and the message does NOT start a new OS/SA topic → REFINE
3. If the message asks about an OS/SA concept in any phrasing → SEARCH
4. If the message is clearly about a non-OS/SA topic and has no history → OUT_OF_SCOPE

Respond with ONLY the single category word: SEARCH, REFINE, CHAT, or OUT_OF_SCOPE.

{history_context}Latest Student Input: "{query}"
"""

REWRITER_PROMPT = """You are a query rewriter for an Operating Systems and System Administration tutor.
Students may write informally, with slang, abbreviations, typos, or incomplete sentences. Your job is to translate any student input into clean, precise, academic search queries suitable for a vector database of OS lecture slides.

RULES:
1. ALWAYS keep the query within the domain of Operating Systems and System Administration. Never introduce topics like databases, web development, or machine learning.
2. Preserve ALL technical concepts from the student's input — do not drop keywords.
3. Fix informal language: expand abbreviations (e.g., "pcb" → "Process Control Block"), fix typos, and make the query formal.
4. If the question is BROAD or AMBIGUOUS (covers multiple sub-topics), split it into TWO specific sub-queries that together cover the full topic.
5. If the question is SPECIFIC and narrow, output ONE query only.
6. Do not add information the student didn't ask about.

Examples:
  Input: "wtf is a semaphore" → QUERY1: What is a semaphore in operating systems and how is it used for process synchronisation?
  Input: "pls explain pcb" → QUERY1: What is a Process Control Block (PCB) and what information does it contain?
  Input: "explain memory management" → QUERY1: What are the memory management techniques used in operating systems such as paging and segmentation? / QUERY2: How does virtual memory and memory allocation work in operating systems?
  Input: "i dont get deadlocks can u explain what conditions needed" → QUERY1: What are the four necessary conditions for a deadlock to occur in an operating system?
  Input: "diff between process and thread" → QUERY1: What is the difference between a process and a thread in terms of memory, execution, and resource sharing?

Respond EXACTLY in this format:
QUERY1: [Your first rewritten query]
QUERY2: [Your second rewritten query only if the topic is broad — otherwise omit this line entirely]

{history_context}Latest Student Input: "{query}"
"""

def get_llm_client():
    """Initialize the LLM client."""
    global _llm_client, _llm_provider
    if _llm_client is not None:
        return _llm_client, _llm_provider

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        print("Using Groq API for LLM reasoning...")
        _llm_client = Groq(api_key=groq_key)
        _llm_provider = "groq"
    else:
        raise ValueError("GROQ_API_KEY not found in .env!")

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


def _format_history(history):
    history_context = ""
    if history:
        history_context = "Conversation History:\n"
        for user_msg, bot_msg in history[-3:]:
            if bot_msg:
                history_context += f"Student: {user_msg}\nTutor: {bot_msg[:150]}...\n"
            else:
                history_context += f"Student: {user_msg}\n"
        history_context += "\n"
    return history_context


def rewrite_query(query, history=None):
    """
    Analyzes student query with context of history.
    Returns: (decision, rewritten_queries, chat_response)
    decision: "SEARCH", "REFINE", "CHAT", or "OUT_OF_SCOPE"
    """
    client, provider = get_llm_client()
    history_context = _format_history(history)

    def _call_llm(prompt):
        if provider == "groq":
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return completion.choices[0].message.content.strip()
        return ""

    try:
        # STEP 1: ROUTER
        router_prompt = ROUTER_PROMPT.format(history_context=history_context, query=query)
        decision_raw = _call_llm(router_prompt)
        
        # Clean the decision
        decision = "SEARCH"
        for valid in ["SEARCH", "REFINE", "CHAT", "OUT_OF_SCOPE"]:
            if valid in decision_raw.upper():
                decision = valid
                break

        # STEP 2: HANDLE DECISION
        if decision == "CHAT":
            return "CHAT", [query], "Hello! 👋 Feel free to ask me anything about Operating Systems & System Administration!"
            
        elif decision == "OUT_OF_SCOPE":
            return "OUT_OF_SCOPE", [query], "I'm sorry, I can only help with topics related to **Operating Systems and System Administration**. That question is outside my scope! Try asking about processes, memory management, file systems, scheduling, or any other OS concept. 🎓"
            
        elif decision == "REFINE":
            if not history:
                decision = "SEARCH"
            else:
                return "REFINE", [query], ""
                
        # STEP 3: REWRITER (If SEARCH)
        if decision == "SEARCH":
            rewriter_prompt = REWRITER_PROMPT.format(history_context=history_context, query=query)
            response_text = _call_llm(rewriter_prompt)
            
            rewritten_queries = []
            for line in response_text.split("\n"):
                if line.startswith("QUERY1:"):
                    q1 = line.replace("QUERY1:", "").strip()
                    if q1 and q1 != "[Your first rewritten query]":
                        rewritten_queries.append(q1)
                elif line.startswith("QUERY2:"):
                    q2 = line.replace("QUERY2:", "").strip()
                    if q2 and q2 != "[Your second rewritten query if the topic is broad/ambiguous, otherwise omit this line]":
                        rewritten_queries.append(q2)
            
            # Clean quotes if model wrapped them
            rewritten_queries = [q[1:-1] if q.startswith('"') and q.endswith('"') else q for q in rewritten_queries]
            
            if not rewritten_queries:
                rewritten_queries = [query]
                
            print(f"User Input: '{query}' -> Decision: SEARCH | Rewritten: {rewritten_queries}")
            return "SEARCH", rewritten_queries, ""

    except Exception as e:
        print(f"Warning: Query rewrite failed ({e}). Defaulting to OUT_OF_SCOPE for safety.")
        return "OUT_OF_SCOPE", [query], "I'm sorry, I encountered an issue processing your request. Please try again or ask a question about Operating Systems and System Administration."


def retrieve_chunks(query, n_results=5):
    """Embed the query and retrieve top-K matching chunks from ChromaDB."""
    openai_ef = get_openai_ef()

    # 1. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    # 2. Query ChromaDB — internally calls OpenAI embeddings API.
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

    if decision in ["CHAT", "OUT_OF_SCOPE", "REFINE"]:
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
