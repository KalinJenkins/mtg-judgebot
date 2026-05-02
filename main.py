# main.py
# MTG Judgebot — Gradio chat UI
# Retrieves relevant rules chunks and uses Claude to answer MTG rules questions.
# Card lookups are handled via the Scryfall API when card names are detected.

import os
from dotenv import load_dotenv
import anthropic
import gradio as gr
from retrieval import load_retriever, retrieve
from scryfall import lookup_cards_in_question, format_card_for_context

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
NUM_RETRIEVAL_RESULTS = 5

SYSTEM_PROMPT = """You are MTG Judgebot, an expert Magic: The Gathering rules assistant.
You answer rules questions clearly and accurately, covering Standard and Commander formats.

You will be given relevant excerpts from official MTG rules documents before each question.
You may also be given card data retrieved from Scryfall. If card data is provided, always
mention that you looked up the card on Scryfall so the player knows where that information
came from.

Base your answers on the provided rules excerpts and card data. Always cite sources.

Guidelines:
- Be precise but approachable, players range from casual to competitive
- If a rule applies differently in Commander vs Standard, explain both
- If the retrieved context doesn't contain enough information to answer confidently, say so
- Format rules citations as: [Source: <filename>, Rule <number>]
- Format card citations as: [Card data: Scryfall]
- Never invent rules or card text — if you're unsure, say so and suggest the player consult a judge"""

# ── Load retriever once at startup ────────────────────────────────────────────

print("Loading retriever...")
model, collection = load_retriever()
print("Retriever ready.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Core logic ────────────────────────────────────────────────────────────────

def build_rules_context(chunks: list[dict]) -> str:
    """Format retrieved rules chunks into a context block for the prompt."""
    lines = ["Relevant rules excerpts:\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Excerpt {i}]")
        lines.append(f"Source: {chunk['source']} | Rule: {chunk['rule_number']}")
        lines.append(chunk['text'])
        lines.append("")
    return "\n".join(lines)


def build_card_context(cards: list[dict]) -> str:
    """Format Scryfall card data into a context block for the prompt."""
    if not cards:
        return ""
    lines = ["Card data from Scryfall:\n"]
    for card in cards:
        lines.append(format_card_for_context(card))
        lines.append("")
    return "\n".join(lines)


def answer_question(query: str, history: list) -> str:
    """
    1. Look up any card names detected in the question via Scryfall
    2. Retrieve relevant rules chunks from ChromaDB
    3. Send both as context to Claude
    """

    # Card lookup
    cards = lookup_cards_in_question(query)
    card_context = build_card_context(cards)
    if cards:
        print(f"  Scryfall: found {len(cards)} card(s): {[c['name'] for c in cards]}")
    else:
        print(f"  Scryfall: no cards detected")

    # Rules retrieval
    chunks = retrieve(query, model, collection)
    rules_context = build_rules_context(chunks)

    # Combine context blocks
    full_context = ""
    if card_context:
        full_context += card_context + "\n"
    full_context += rules_context

    # Build conversation history for Claude
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
        "content": f"{full_context}\n\nQuestion: {query}"
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
    gr.Markdown("# MTG Judgebot")
    gr.Markdown("Ask rules questions about **Standard** or **Commander** format.")
    gr.Markdown("""
    Answers are sourced from the official MTG Comprehensive Rules, Magic Tournament Rules, 
    and Commander format rules. Card data is retrieved live from the Scryfall API.
    ⚠️ Ban lists are not included — always verify card legality before a tournament.
    [View on GitHub](https://github.com/KalinJenkins/mtg-judgebot)
        """)


    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height="70vh"),
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
