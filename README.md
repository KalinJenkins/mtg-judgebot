# MTG Judgebot

A Magic: The Gathering rules assistant powered by RAG (Retrieval-Augmented Generation). Ask natural language rules questions and get answers sourced directly from official rules documents, with citations.

Built as a learning project to explore RAG on a complex, structured domain, the MTG Comprehensive Rules are 300+ pages of dense, interconnected rules, making them a more interesting retrieval problem than typical document Q&A.

## Example

```
Player: What happens if my commander dies?

Judgebot: When your commander would go to the graveyard, you have the option to
move it to the command zone instead. This is a replacement effect — you choose
whether to apply it each time the situation arises.

[Source: commander-rules.txt, Rule chunk_1]
[Source: comprehensive-rules.txt, Rule 903]
```

## Architecture

```
User Question
     │
     ▼
Embedding Model (all-MiniLM-L6-v2)
     │
     ▼
ChromaDB Vector Search → Top 5 relevant rules chunks
     │
     ▼
Claude API (context + question)
     │
     ▼
Answer with citations
```

### Card lookup via Scryfall
When a question mentions a card name, Judgebot queries the Scryfall API for that card's oracle text, mana cost, type, and format legality. Scryfall's fuzzy matching handles misspellings and missing punctuation. Card data is combined with the rules context before being sent to Claude.


## Stack

- Python 3.12
- [Anthropic Claude API](https://anthropic.com) — answer generation
- [ChromaDB](https://www.trychroma.com/) — local vector database
- [sentence-transformers](https://www.sbert.net/) — embeddings (`all-MiniLM-L6-v2`)
- [Gradio](https://www.gradio.app/) — chat UI
- [pypdf](https://pypdf.readthedocs.io/) — PDF text extraction
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — HTML stripping
- [Scryfall API](https://scryfall.com/docs/api) — live card data lookup, no API key required

## Setup

**Prerequisites:** Python 3.10+, an Anthropic API key

```bash
git clone git@github.com:KalinJenkins/mtg-judgebot.git
cd mtg-judgebot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Download the rules documents into `rules/`:
```bash
mkdir rules
curl -o rules/comprehensive-rules.txt "https://media.wizards.com/2026/downloads/MagicCompRules%2020260227.txt"
curl -o rules/tournament-rules.pdf "https://media.wizards.com/ContentResources/WPN/MTG_MTR_2026_Feb27_EN.pdf"
```

For the Commander rules, manually copy the text from the following pages into `rules/commander-rules.txt`:
- https://mtgcommander.net/index.php/rules/
- https://mtgcommander.net/index.php/the-philosophy-of-commander/
- https://mtgcommander.net/index.php/faq/

Build the vector database:
```bash
python3 ingest.py
```

Run the app:
```bash
python3 main.py
```

Open `http://localhost:7860` in your browser.

## Updating rules

MTG rules update approximately four times per year with each major set release. To update:

1. Download the new rules documents into `rules/` using the URLs above (update the date in the filename)
2. Re-run `python3 ingest.py` — this wipes and rebuilds the vector database from scratch

## Known limitations

- **Ban lists not included** — Standard and Commander ban lists change frequently and are not ingested. Judgebot will not have accurate information about whether specific cards are currently banned.
- **Rules currency** — Answers are only as current as the last ingestion. If rules have been updated since the database was built, Judgebot may give outdated answers.
- **Commander rules source** — The Commander rules are manually copied plain text from mtgcommander.net rather than a structured official document, which means they have less precise citation metadata than the Comprehensive Rules.
- **Retrieval limits** — Complex multi-rule interactions may require context from several different rule sections simultaneously. If the relevant rules don't surface in the top 5 retrieved chunks, the answer may be incomplete.
