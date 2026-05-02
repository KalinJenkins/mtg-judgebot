# ingest.py
# Loads MTG rules documents into a ChromaDB vector database.
# Handles: .txt files (plain text or HTML), .pdf files
# Uses rule-number-aware chunking for MTG structured content.

import os
import re
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────

RULES_DIR = "rules"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "mtg_rules"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Fallback chunk size (characters) for documents without MTG rule numbering
GENERIC_CHUNK_SIZE = 1000
GENERIC_CHUNK_OVERLAP = 150

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(filepath: str) -> str:
    """
    Extract text from a .txt file.
    If the file contains HTML (e.g. a saved webpage), strip the tags.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Detect HTML by looking for common tags
    if "<html" in raw.lower() or "<body" in raw.lower() or "<div" in raw.lower():
        soup = BeautifulSoup(raw, "html.parser")
        # Remove nav, header, footer, script, style — we only want content
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    return raw


def extract_text(filepath: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        print(f"  Skipping unsupported file type: {filepath}")
        return ""

# ── Chunking ──────────────────────────────────────────────────────────────────

# Matches MTG rule numbers like: 100.  702.15a  100.1.  903.
MTG_RULE_PATTERN = re.compile(r"^\d{3,}\.\d*[a-z]?\s", re.MULTILINE)


def is_mtg_structured(text: str) -> bool:
    """Return True if the text looks like MTG Comprehensive Rules."""
    matches = MTG_RULE_PATTERN.findall(text)
    return len(matches) > 50  # Threshold: at least 50 rule numbers found


def chunk_by_rule_number(text: str, source_name: str) -> list[dict]:
    """
    Split text at MTG rule number boundaries.
    Each chunk groups related sub-rules together (e.g. all of 702.15 + 702.15a/b/c).
    Returns a list of dicts with 'text', 'source', and 'rule_number' fields.
    """
    # Find all positions where a top-level rule starts (e.g. "702. " or "702.15 ")
    # We group by the first number before the dot to keep related rules together
    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_rule = None

    for line in lines:
        match = MTG_RULE_PATTERN.match(line)
        if match:
            # Extract the top-level rule number (e.g. "702" from "702.15a")
            rule_num = line.split(".")[0].strip()

            if current_rule is not None and rule_num != current_rule:
                # We've moved to a new top-level rule — save the previous chunk
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": source_name,
                        "rule_number": current_rule,
                    })
                current_chunk_lines = []

            current_rule = rule_num

        current_chunk_lines.append(line)

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "rule_number": current_rule or "unknown",
            })

    return chunks


def chunk_generic(text: str, source_name: str) -> list[dict]:
    """
    Generic character-based chunking with overlap.
    Used for documents without MTG rule numbering (MTR, Commander rules).
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + GENERIC_CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "rule_number": f"chunk_{chunk_index}",
            })

        start += GENERIC_CHUNK_SIZE - GENERIC_CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def chunk_document(text: str, source_name: str) -> list[dict]:
    """Choose the right chunking strategy for this document."""
    if is_mtg_structured(text):
        print(f"  Detected MTG rule structure — using rule-number chunking")
        return chunk_by_rule_number(text, source_name)
    else:
        print(f"  Using generic chunking")
        return chunk_generic(text, source_name)

# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest():
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    # Set up ChromaDB
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Process each file in the rules directory
    rules_path = Path(RULES_DIR)
    all_files = list(rules_path.iterdir())

    total_chunks = 0

    for filepath in sorted(all_files):
        if filepath.suffix.lower() not in (".txt", ".pdf"):
            continue

        print(f"\nProcessing: {filepath.name}")
        text = extract_text(str(filepath))

        if not text.strip():
            print(f"  No text extracted — skipping")
            continue

        print(f"  Extracted {len(text):,} characters")

        chunks = chunk_document(text, filepath.name)
        print(f"  Created {len(chunks)} chunks")

        if not chunks:
            continue

        # Embed and store in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts).tolist()

            collection.add(
                ids=[f"{filepath.stem}_{i + j}" for j, _ in enumerate(batch)],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    "source": c["source"],
                    "rule_number": c["rule_number"],
                } for c in batch],
            )

        total_chunks += len(chunks)
        print(f"  Stored {len(chunks)} chunks in ChromaDB")

    print(f"\nIngestion complete. Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    ingest()
