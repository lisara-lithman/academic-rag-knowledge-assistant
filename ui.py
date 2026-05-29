import os
import gradio as gr
import uuid
from dotenv import load_dotenv
from google.genai import types
from retrieval import search_pipeline, get_llm_client

load_dotenv()

# ── Custom CSS ────────────────────────────────────────────────────────────────
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(to right, #60a5fa, #3b82f6, #1d4ed8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

# ── Per-session chunk store keyed by a stable UUID ───────────────────────────
# gr.State holds the UUID; this dict holds the actual chunks.
_chunk_store: dict = {}


# ── Answer generation ─────────────────────────────────────────────────────────
def generate_grounded_answer(query, context_chunks, history=None, is_refinement=False):
    client, provider = get_llm_client()

    context_text = ""
    for i, chunk in enumerate(context_chunks):
        src = chunk['metadata']['source']
        pg  = chunk['metadata']['page']
        context_text += f"--- Slide {i+1} (Source: {src}, Page {pg}) ---\n{chunk['text']}\n\n"

    system_prompt = (
        "You are an expert, friendly AI teaching assistant for the university module "
        "'Operating Systems and System Administration'.\n\n"
        "Answer the student's question using ONLY the provided lecture slides. Rules:\n"
        "1. Base your answer strictly on the context — no outside knowledge.\n"
        "2. If the context lacks information, say: 'I cannot find the answer in the module materials.'\n"
        "3. Give a clear, structured explanation with bullet points and bold text where helpful.\n"
        "4. Do NOT mention file names or page numbers inside your answer.\n"
        "5. If asked for a diagram, use Mermaid.js (```mermaid blocks) or ASCII art."
    )

    if is_refinement:
        system_prompt += (
            "\n\nIMPORTANT: The student wants you to refine or shorten the previous answer. "
            "Fulfill their request using the conversation history — do not start fresh."
        )

    user_prompt = (
        f"Context Materials:\n{context_text}\n\n"
        f"Student Question: \"{query}\"\n"
        "Answer using ONLY the context materials above."
    )

    if provider == "groq":
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for u, b in history:
                if u: messages.append({"role": "user",      "content": u})
                if b: messages.append({"role": "assistant", "content": b})
        messages.append({"role": "user", "content": user_prompt})
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()

    elif provider == "gemini":
        contents = []
        if history:
            for u, b in history:
                if u: contents.append(types.Content(role="user",  parts=[types.Part.from_text(text=u)]))
                if b: contents.append(types.Content(role="model", parts=[types.Part.from_text(text=b)]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
        config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.2)
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=contents, config=config
        )
        return response.text.strip()

    return "Error: unknown provider."


# ── Gradio handlers ───────────────────────────────────────────────────────────
def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history, session_id):
    """Run RAG pipeline. session_id (gr.State UUID) is stable per browser tab."""
    if not history:
        return history, "<p style='color:gray;'>Ask a question to see the references.</p>", session_id

    user_message = history[-1][0]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        decision, _, new_chunks = search_pipeline(user_message, history=history[:-1])
    except Exception as e:
        print(f"[ERROR] search_pipeline: {e}")
        history[-1][1] = f"⚠️ Retrieval error: {e}"
        return history, "<p style='color:red;'>Retrieval failed.</p>", session_id

    # Keep chunks across turns using the stable session UUID
    if new_chunks:
        _chunk_store[session_id] = new_chunks
        chunks = new_chunks
    else:
        chunks = _chunk_store.get(session_id, [])

    # ── Answer ────────────────────────────────────────────────────────────────
    is_refinement = (decision == "REFINE")
    try:
        answer = generate_grounded_answer(
            user_message, chunks,
            history=history[:-1],
            is_refinement=is_refinement
        )
    except Exception as e:
        print(f"[ERROR] generate_grounded_answer: {e}")
        answer = f"⚠️ Answer generation error: {e}"

    history[-1][1] = answer

    # ── Sources panel ─────────────────────────────────────────────────────────
    if not chunks:
        sources_html = "<p style='color:#64748b;font-style:italic;'>No reference slides found for this query.</p>"
    else:
        sources_html = "<h3 style='margin-top:0;color:#1e3a8a;border-bottom:2px solid #e2e8f0;padding-bottom:8px;'>📚 Sources Used</h3>"
        for chunk in chunks:
            src     = chunk['metadata'].get('source', 'Unknown')
            pg      = chunk['metadata'].get('page', '?')
            score   = chunk.get('rerank_score', 0.0)
            snippet = chunk['text'][:120].replace('\n', ' ') + "…"
            sources_html += f"""
            <div style="background:#f8fafc;border-left:4px solid #3b82f6;
                        padding:12px;margin-bottom:10px;border-radius:6px;border:1px solid #e2e8f0;">
                <div style="font-weight:700;color:#1e293b;font-size:13px;">{src} — Page {pg}</div>
                <div style="font-size:11px;color:#3b82f6;font-weight:600;margin:4px 0;">Relevance: {score:.3f}</div>
                <div style="font-style:italic;color:#475569;font-size:12px;
                            border-top:1px dashed #e2e8f0;padding-top:6px;margin-top:6px;">"{snippet}"</div>
            </div>
            """

    return history, sources_html, session_id


def clear_chat(session_id):
    _chunk_store.pop(session_id, None)
    placeholder = "<p style='color:#64748b;font-style:italic;'>Ask a question to view slides and page citations.</p>"
    return [], placeholder, session_id


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css=custom_css
) as demo:

    gr.HTML("""
        <div style="text-align:center;margin:24px 0 16px 0;">
            <h1>🎓 Academic RAG Knowledge Assistant</h1>
            <p style="color:#475569;font-size:1.05rem;margin-top:4px;">
                Interactive AI Tutor for
                <b>Operating Systems &amp; System Administration</b>
            </p>
        </div>
    """)

    # Stable UUID per browser session — generated once when the page loads
    session_id = gr.State(lambda: str(uuid.uuid4()))

    with gr.Row(equal_height=True):

        # ── Left: chat ────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about scheduling, deadlocks, memory management…",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

        # ── Right: sources ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            sources_panel = gr.HTML(
                value="<p style='color:#64748b;font-style:italic;'>Ask a question to view the slides and page citations used to formulate the response.</p>"
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    # Step 1: append user message (fast, no queue needed)
    # Step 2: run RAG pipeline (queued so only one runs at a time)
    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot, [chatbot, session_id], [chatbot, sources_panel, session_id]
    )

    submit_btn.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot, [chatbot, session_id], [chatbot, sources_panel, session_id]
    )

    clear_btn.click(
        clear_chat, [session_id], [chatbot, sources_panel, session_id], queue=False
    )


if __name__ == "__main__":
    print("Launching OSSA AI Knowledge Assistant UI…")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
