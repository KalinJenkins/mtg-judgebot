# scryfall.py
# Handles all Scryfall API calls for MTG card lookups.
# Scryfall API docs: https://scryfall.com/docs/api
# No API key required. Please keep requests reasonable per Scryfall's guidelines.

import requests
import time

SCRYFALL_BASE = "https://api.scryfall.com"

# Scryfall asks for a small delay between requests
REQUEST_DELAY = 0.1


def get_card_by_name(name: str) -> dict | None:
    """
    Look up a card by name using Scryfall's fuzzy matching.
    Returns a dict of card data, or None if not found.
    """
    time.sleep(REQUEST_DELAY)

    response = requests.get(
        f"{SCRYFALL_BASE}/cards/named",
        params={"fuzzy": name},
        timeout=10,
    )

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        print(f"Scryfall error {response.status_code}: {response.text}")
        return None


def format_card_for_context(card: dict) -> str:
    """
    Format a Scryfall card object into a readable block for Claude's context.
    Handles normal cards and double-faced cards.
    """
    lines = []
    lines.append(f"Card: {card.get('name')}")
    lines.append(f"Mana Cost: {card.get('mana_cost', 'N/A')}")
    lines.append(f"Type: {card.get('type_line', 'N/A')}")

    # Double-faced cards store oracle text per face
    if "card_faces" in card:
        for face in card["card_faces"]:
            lines.append(f"  Face: {face.get('name')}")
            lines.append(f"  Type: {face.get('type_line', 'N/A')}")
            if face.get("oracle_text"):
                lines.append(f"  Text: {face.get('oracle_text')}")
            if face.get("power"):
                lines.append(f"  P/T: {face.get('power')}/{face.get('toughness')}")
    else:
        if card.get("oracle_text"):
            lines.append(f"Oracle Text: {card.get('oracle_text')}")
        if card.get("power"):
            lines.append(f"P/T: {card.get('power')}/{card.get('toughness')}")

    if card.get("loyalty"):
        lines.append(f"Loyalty: {card.get('loyalty')}")

    lines.append(f"Legal in Standard: {card.get('legalities', {}).get('standard', 'unknown')}")
    lines.append(f"Legal in Commander: {card.get('legalities', {}).get('commander', 'unknown')}")

    return "\n".join(lines)


def lookup_cards_in_question(question: str) -> list[dict]:
    """
    Attempt to find card names in a question using Scryfall's fuzzy search.
    Tries progressively larger word windows (2, 3, 4, 5 words) looking for matches.
    Returns a list of successfully matched card dicts.
    """
    words = question.split()
    found_cards = []
    found_names = set()

    # Try windows of 2–5 words to catch multi-word card names
    for window_size in range(5, 1, -1):
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i:i + window_size])
            # Skip if we already matched something overlapping
            if any(candidate.lower() in name.lower() or name.lower() in candidate.lower()
                   for name in found_names):
                continue

            card = get_card_by_name(candidate)
            if card:
                found_cards.append(card)
                found_names.add(card["name"])

    return found_cards


if __name__ == "__main__":
    # Quick standalone test
    test_names = [
        "Sol Ring",
        "sol ring",           # lowercase
        "Lightnig Bolt",      # deliberate misspelling
        "Teferi Hero of Dominaria",  # missing comma
        "definitely not a real card",
    ]

    for name in test_names:
        print(f"\nLooking up: '{name}'")
        card = get_card_by_name(name)
        if card:
            print(format_card_for_context(card))
        else:
            print("  Not found")
