"""
Step 2: Convert enriched JSONL into 3 training formats:
  A) RAG embeddings  — ChromaDB per character
  B) Few-shot examples — for system prompt injection
  C) SFT format  — for optional local fine-tuning with Ollama/LLaMA
"""

import json
import os
from collections import defaultdict

def load_enriched(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ── FORMAT A: ChromaDB documents ──────────────────────────────────────────────
def build_chroma_docs(records):
    """
    Each record becomes a ChromaDB document.
    Document = the quote itself (what gets embedded + retrieved)
    Metadata = everything else (filtered at query time)
    """
    by_character = defaultdict(list)
    for r in records:
        doc = {
            "id": f"{r['speaker']}_{hash(r['quote']) % 999999}",
            "document": r['quote'],  # embedded text
            "metadata": {
                "speaker": r["speaker"],
                "emotion": r["emotion"],
                "intent": r["intent"],
                "defense_mechanism": r["defense_mechanism"],
                "topic": r["topic"],
                "responding_to": r["responding_to"],
                "to_whom": r["to_whom"],
                "context": r["context_window"]
            }
        }
        by_character[r["speaker"]].append(doc)

    for char, docs in by_character.items():
        out_path = f"chroma_docs_{char.lower()}.jsonl"
        with open(out_path, 'w') as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        print(f"[ChromaDB] {char}: {len(docs)} documents → {out_path}")

    return by_character

# ── FORMAT B: Few-shot examples (top examples per intent/emotion combo) ────────
def build_few_shot_examples(records):
    """
    Select best examples per character per (emotion, intent) combination.
    These go into the system prompt as demonstrations.
    """
    by_char_combo = defaultdict(list)
    for r in records:
        key = (r["speaker"], r["emotion"], r["intent"])
        by_char_combo[key].append(r)

    by_character = defaultdict(list)
    for (char, emotion, intent), examples in by_char_combo.items():
        # Pick the longest / richest quote for each combo
        best = max(examples, key=lambda x: len(x["quote"]))
        by_character[char].append({
            "situation": f"When feeling {emotion}, intent to {intent}",
            "context": best["context_window"],
            "response": best["quote"],
            "defense_mechanism": best["defense_mechanism"]
        })

    for char, examples in by_character.items():
        out_path = f"few_shot_{char.lower()}.json"
        with open(out_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"[Few-shot] {char}: {len(examples)} examples → {out_path}")

    return by_character

# ── FORMAT C: SFT (Supervised Fine-Tuning) format for Ollama/LLaMA ───────────
SFT_SYSTEM = """You are {character}. You must respond exactly as {character} from Friends would — with their speech patterns, humor, defense mechanisms, and worldview. You never break character. You never use AI-assistant language.

Character facts:
- Profession: {profession}
- Core fear: {core_fear}
- Personality: {personality}
- Known topics: {known_topics}

Stay within your character's knowledge. If asked something outside your world, respond in-character with confusion or deflection."""

CHARACTER_META = {
    "Chandler": {"core_fear": "vulnerability", "personality": "sarcastic, anxious, secretly kind"},
    "Joey":     {"core_fear": "being seen as stupid", "personality": "loyal, food-obsessed, charming"},
    "Monica":   {"core_fear": "losing control", "personality": "competitive, nurturing, perfectionist"},
    "Ross":     {"core_fear": "being unloved", "personality": "intellectual, romantic, pedantic"},
    "Phoebe":   {"core_fear": "abandonment", "personality": "spiritual, optimistic, sincerely strange"},
    "Rachel":   {"core_fear": "becoming her mother", "personality": "charming, growth-oriented, fashion-focused"}
}

PROFESSIONS = {
    "Chandler": "Statistical analysis and data reconfiguration",
    "Joey":     "Actor",
    "Monica":   "Chef",
    "Ross":     "Paleontologist and Professor",
    "Phoebe":   "Masseuse and Musician",
    "Rachel":   "Fashion industry (Ralph Lauren)"
}

KNOWN_TOPICS = {
    "Chandler": "office, sarcasm, relationships, pop culture, self-deprecation",
    "Joey":     "food, acting, women, sports, sandwiches",
    "Monica":   "cooking, cleaning, competitions, family, organization",
    "Ross":     "dinosaurs, academia, relationships, science, museums",
    "Phoebe":   "spirits, massage, music, nature, past lives, animals",
    "Rachel":   "fashion, relationships, shopping, career, independence"
}

def build_sft_data(records):
    """
    Build (system, user, assistant) triples for SFT.
    User = the context/trigger. Assistant = the character's line.
    """
    by_character = defaultdict(list)
    for r in records:
        char = r["speaker"]
        if char not in CHARACTER_META:
            continue
        meta = CHARACTER_META[char]
        system = SFT_SYSTEM.format(
            character=char,
            profession=PROFESSIONS.get(char, ""),
            core_fear=meta["core_fear"],
            personality=meta["personality"],
            known_topics=KNOWN_TOPICS.get(char, "")
        )
        # User turn = the context that triggered this line
        user_turn = r["context_window"] if r["context_window"] != "Start of scene" \
                    else f"You're hanging out with your friends in New York."

        sft_record = {
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user_turn},
                {"role": "assistant", "content": r["quote"]}
            ],
            "metadata": {
                "emotion": r["emotion"],
                "intent": r["intent"],
                "defense_mechanism": r["defense_mechanism"],
                "topic": r["topic"]
            }
        }
        by_character[char].append(sft_record)

    for char, items in by_character.items():
        out_path = f"sft_{char.lower()}.jsonl"
        with open(out_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"[SFT] {char}: {len(items)} training pairs → {out_path}")

    return by_character

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    enriched_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_dialogues.jsonl"

    if not os.path.exists(enriched_path):
        print(f"ERROR: {enriched_path} not found. Run enrich_dialogues.py first.")
        exit(1)

    records = load_enriched(enriched_path)
    print(f"Loaded {len(records)} enriched records\n")

    print("── Building ChromaDB documents ──")
    build_chroma_docs(records)

    print("\n── Building few-shot examples ──")
    build_few_shot_examples(records)

    print("\n── Building SFT training data ──")
    build_sft_data(records)

    print("\nAll formats built. Next step: run chroma_ingest.py")
