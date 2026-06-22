import os
import gradio as gr
from dotenv import load_dotenv
from google.genai import types
from retrieval import search_pipeline, get_llm_client, get_openai_ef, load_rerank_model, call_with_retry

load_dotenv()

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

def stream_grounded_answer(query, context_chunks, history=None, is_refinement=False):
    """Stream the grounded answer token by token using the LLM's streaming API."""
    client, provider = get_llm_client()

    context_text = ""
    for i, chunk in enumerate(context_chunks):
        src = chunk['metadata']['source']
        pg  = chunk['metadata']['page']
        context_text += f"--- Slide {i+1} (Source: {src}, Page {pg}) ---\n{chunk['text']}\n\n"

    system_prompt = (
        "You are an expert, friendly AI teaching assistant for the university module "
        "'Operating Systems and System Administration'.\n\n"
        "Rules for answering the student's question:\n"
        "1. Prioritize using the provided lecture slides as your main source of truth.\n"
        "2. If the context materials lack the necessary information, you MAY use your own "
        "expert knowledge of Operating Systems to answer the question. However, you MUST explicitly state: "
        "'*The lecture slides do not fully cover this, but generally speaking...*' so the student knows it's outside material.\n"
        "3. If the student asks about a topic COMPLETELY UNRELATED to Operating Systems and System Administration "
        "(e.g., Web Development, Node.js, Cooking, pop culture), you MUST politely refuse to answer and say: "
        "'I can only help with topics related to Operating Systems.'\n"
        "4. ALWAYS start by directly answering the question in 1-2 sentences before expanding.\n"
        "5. Match your length and tone to the student's question:\n"
        "   - Casual questions → plain English, 2-4 sentences max.\n"
        "   - Formal questions → use structured bullet points and bold key terms.\n"
        "6. Do NOT mention file names or page numbers inside your answer.\n"
        "7. If asked for a diagram, use Mermaid.js (```mermaid blocks) or ASCII art."
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

    if provider == "openai":
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for u, b in history:
                if u: messages.append({"role": "user",      "content": u})
                if b: messages.append({"role": "assistant", "content": b})
        messages.append({"role": "user", "content": user_prompt})
        stream = call_with_retry(
            lambda: client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.2, stream=True
            )
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def generate_grounded_answer(query, context_chunks, history=None, is_refinement=False):
    """Non-streaming wrapper around stream_grounded_answer for batch evaluation."""
    return "".join(
        stream_grounded_answer(query, context_chunks, history=history, is_refinement=is_refinement)
    )


def respond(user_message, history, last_chunks):
    if not user_message.strip():
        yield "", history, "<p style='color:#64748b;font-style:italic;'>Ask a question to view slides and citations.</p>", last_chunks
        return

    try:
        decision, _, new_chunks, chat_response = search_pipeline(user_message, history=history)
    except Exception as e:
        print(f"[ERROR] search_pipeline: {e}")
        history = history + [(user_message, f"⚠️ Retrieval error: {e}")]
        yield "", history, "<p style='color:red;'>Retrieval failed.</p>", last_chunks
        return

    # Handle greetings and purely conversational messages — the router already
    # generated a natural reply in the same LLM call, so use it directly.
    if decision == "CHAT":
        history = history + [(user_message, chat_response)]
        no_sources = "<p style='color:#64748b;font-style:italic;'>No slides needed for this response.</p>"
        yield "", history, no_sources, last_chunks
        return

    if new_chunks:
        chunks = new_chunks
    elif decision == "SEARCH":
        # SEARCH was performed but every chunk scored below the relevance
        # threshold. Pass empty chunks to the LLM so it can answer using its own knowledge.
        chunks = []
    else:
        chunks = last_chunks

    # Build sources HTML (shown immediately, before streaming begins)
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

    is_refinement = (decision == "REFINE")

    # Stream the answer token by token.
    # context_history is the history WITHOUT the current exchange (for LLM multi-turn context).
    context_history = list(history)
    history = history + [(user_message, "")]

    try:
        for token in stream_grounded_answer(user_message, chunks, history=context_history, is_refinement=is_refinement):
            history[-1] = (history[-1][0], history[-1][1] + token)
            yield "", history, sources_html, chunks
    except Exception as e:
        print(f"[ERROR] stream_grounded_answer: {e}")
        history[-1] = (history[-1][0], f"⚠️ Answer generation error: {e}")
        yield "", history, sources_html, chunks


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

    last_chunks = gr.State([])

    with gr.Row(equal_height=True):
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

        with gr.Column(scale=1):
            sources_panel = gr.HTML(
                value="<p style='color:#64748b;font-style:italic;'>Ask a question to view the slides and page citations used to formulate the response.</p>"
            )

    inputs  = [msg, chatbot, last_chunks]
    outputs = [msg, chatbot, sources_panel, last_chunks]

    msg.submit(respond, inputs, outputs)
    submit_btn.click(respond, inputs, outputs)
    
    def clear_chat():
        return "", [], "<p style='color:#64748b;font-style:italic;'>Ask a question to view slides and page citations.</p>", []
        
    clear_btn.click(clear_chat, None, outputs, queue=False)


if __name__ == "__main__":
    print("Pre-loading models into memory — this takes a few seconds...")
    get_openai_ef()
    load_rerank_model()
    print("Models loaded. Launching OSSA AI Knowledge Assistant UI…")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
