import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional

# Import our perfectly functioning backend logic
from retrieval import search_pipeline, get_llm_client, get_openai_ef, load_rerank_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warm up models to ensure fast first-query
    print("Pre-loading models into memory...")
    get_openai_ef()
    load_rerank_model()
    get_llm_client()
    import chromadb
    chromadb.PersistentClient(path="./vector_db")
    
    print("Warming up pipeline...")
    try:
        search_pipeline("Hello", history=[])
    except Exception as e:
        pass
    
    print("Models loaded. Server ready.")
    yield
    # Shutdown
    print("Shutting down server.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow React app (Vite runs on 5173 typically)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = [] # Format: [(user, bot), ...]

def stream_grounded_answer(query, context_chunks, history=None, is_refinement=False):
    """Stream the grounded answer token by token using the LLM's streaming API."""
    client, provider = get_llm_client()
    context_text = "\n\n".join([f"--- Source: {c['metadata'].get('source', 'Unknown')} (Page {c['metadata'].get('page', '?')}) ---\n{c['text']}" for c in context_chunks])
    
    system_prompt = (
        "You are an expert AI tutor for a university 'Operating Systems and System Administration' (OSSA) module. "
        "Your task is to answer the student's question clearly, accurately, and pedagogically.\n\n"
        "RULES:\n"
        "- Base your answer heavily on the provided Context Materials. "
        "- If the Context Materials are insufficient, you may use your internal knowledge of OS concepts to supplement the answer, but keep it accurate to standard university curricula.\n"
        "- Use markdown formatting (bolding, lists, code blocks) to make your explanation readable.\n"
        "- Keep it concise unless a detailed explanation is requested."
    )
    if is_refinement:
        system_prompt += (
            "\n\nIMPORTANT: The student wants you to refine or shorten the previous answer. "
            "Fulfill their request using the conversation history — do not start fresh."
        )

    user_prompt = (
        f"Context Materials:\n{context_text}\n\n"
        f"Student Question: \"{query}\"\n"
        "Answer directly and concisely, prioritizing the context materials but using outside knowledge if necessary."
    )

    if provider == "groq":
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for u, b in history:
                if u: messages.append({"role": "user",      "content": u})
                if b: messages.append({"role": "assistant", "content": b})
        messages.append({"role": "user", "content": user_prompt})
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, temperature=0.2, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message
    history = req.history

    if not user_message.strip():
        return StreamingResponse(iter([]))

    def generate_events():
        # 1. Run Pipeline
        decision, _, new_chunks, chat_response = search_pipeline(user_message, history=history)
        
        # Determine chunks to use
        if new_chunks:
            chunks = new_chunks
        elif decision == "SEARCH":
            chunks = []
        else:
            chunks = [] # If CHAT, we don't need chunks
            
        # Yield metadata first (decision, chunks)
        meta_event = json.dumps({
            "type": "metadata",
            "decision": decision,
            "chunks": chunks
        })
        yield f"data: {meta_event}\n\n"
        
        # 2. Stream Response
        if decision == "CHAT":
            # Just stream the whole chat_response as one token for simplicity
            token_event = json.dumps({"type": "token", "content": chat_response})
            yield f"data: {token_event}\n\n"
        else:
            is_refinement = (decision == "REFINE")
            generator = stream_grounded_answer(user_message, chunks, history=history, is_refinement=is_refinement)
            for token in generator:
                token_event = json.dumps({"type": "token", "content": token})
                yield f"data: {token_event}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate_events(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
