# main.py
# MTG Judgebot — Gradio chat UI
# Retrieves relevant rules chunks and uses Claude to answer MTG rules questions.

import os
from dotenv import load_dotenv
import anthropic
import gradio as gr
from retrieval import load_retriever, retrieve

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
NUM_RETRIEVAL_RESULTS = 5

SYSTEM_PROMPT = """You are MTG Judgebot, an expert Magic: The Gathering rules assistant.
You answer rules questions clearly and accurately, covering Standard and Commander formats.

You will be given relevant excerpts from official MTG rules documents before each question.
Base your answers on those excerpts. Always cite the source and rule number when available.

Guidelines:
- Be precise but approachable — players range from casual to competitive
- If a rule applies differently in Commander vs Standard, explain both
- If the retrieved context doesn't contain enough information to answer confidently, say so
- Format citations as: [Source: <filename>, Rule <number>]
- Never invent rules — if you're unsure, say so and suggest the player consult a judge"""

# ── Load retriever once at startup ────────────────────────────────────────────

print("Loading retriever...")
model, collection = load_retriever()
print("Retriever ready.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Core logic ────────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    lines = ["Relevant rules excerpts:\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Excerpt {i}]")
        lines.append(f"Source: {chunk['source']} | Rule: {chunk['rule_number']}")
        lines.append(chunk['text'])
        lines.append("")
    return "\n".join(lines)


def answer_question(query: str, history: list) -> str:
    """Retrieve relevant chunks and ask Claude to answer the question."""

    # Retrieve relevant rules chunks
    chunks = retrieve(query, model, collection)
    context = build_context(chunks)

    # Build conversation history for Claude
    # Handle both old tuple format and new Gradio dict format
    messages = []
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, bot_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

    # Add current question with context prepended
    messages.append({
        "role": "user",
        "content": f"{context}\n\nQuestion: {query}"
    })

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def chat(message, history):
    return answer_question(message, history)


with gr.Blocks(title="MTG Judgebot") as demo:
    gr.Markdown("# ⚖️ MTG Judgebot")
    gr.Markdown("Ask rules questions about **Standard** or **Commander** format.")

    chatbot = gr.ChatInterface(
        fn=chat,
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
