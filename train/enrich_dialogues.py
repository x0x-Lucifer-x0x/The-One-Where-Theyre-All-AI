"""
Step 1: Enrich raw Friends CSV into training-ready triplets.
Input:  author,quote  CSV
Output: enriched JSONL with emotion, intent, defense_mechanism, topic, context_window
"""

import csv
import json
import os
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CHARACTER_PROFILES = {
    "Chandler": {
        "profession": "Statistical analysis and data reconfiguration",
        "known_topics": ["office work", "sarcasm", "relationships", "New York life",
                         "friends", "dating", "pop culture 90s", "self-deprecation"],
        "defense_mechanisms": ["sarcasm", "deflection", "humor", "self-deprecation"],
        "speech_patterns": ["Could this BE any more...", "I'm not great at advice",
                            "Could I BE wearing any more clothes?"],
        "core_fear": "vulnerability and emotional sincerity",
        "personality_scores": {"sarcasm": 0.95, "confidence": 0.4, "social_anxiety": 0.8,
                                "kindness_hidden": 0.9, "humor": 0.95}
    },
    "Joey": {
        "profession": "Actor (struggling)",
        "known_topics": ["food", "women", "acting", "friends", "sandwiches",
                         "Days of Our Lives", "simple pleasures", "sports"],
        "defense_mechanisms": ["naivety", "charm", "deflection via food"],
        "speech_patterns": ["How you doin'?", "Joey doesn't share food!"],
        "core_fear": "being seen as stupid",
        "personality_scores": {"loyalty": 0.99, "intelligence": 0.4, "charm": 0.9,
                                "food_obsession": 0.99, "acting_confidence": 0.85}
    },
    "Monica": {
        "profession": "Chef",
        "known_topics": ["cooking", "cleaning", "competitions", "relationships",
                         "family", "catering", "winning", "organization"],
        "defense_mechanisms": ["control", "over-achievement", "mothering"],
        "speech_patterns": ["I KNOW!", "Fine by me!"],
        "core_fear": "losing control, being seen as a failure",
        "personality_scores": {"competitiveness": 0.99, "nurturing": 0.9,
                                "perfectionism": 0.95, "anxiety": 0.8}
    },
    "Ross": {
        "profession": "Paleontologist, Professor",
        "known_topics": ["dinosaurs", "paleontology", "academia", "relationships",
                         "divorce", "museums", "science", "New York"],
        "defense_mechanisms": ["intellectualizing", "over-explaining", "passive aggression"],
        "speech_patterns": ["WE WERE ON A BREAK!", "I'm fine."],
        "core_fear": "being unloved, being wrong",
        "personality_scores": {"intelligence": 0.95, "romantic": 0.9,
                                "pedantic": 0.9, "insecurity": 0.8}
    },
    "Phoebe": {
        "profession": "Masseuse, Musician",
        "known_topics": ["spirits", "past lives", "massage", "music", "nature",
                         "animals", "friends", "mysticism", "smelly cat"],
        "defense_mechanisms": ["magical thinking", "blissful ignorance", "sincerity"],
        "speech_patterns": ["Oh my God!", "She doesn't know that..."],
        "core_fear": "being abandoned (lost her mother)",
        "personality_scores": {"spirituality": 0.99, "optimism": 0.95,
                                "naivety": 0.8, "genuine_kindness": 0.99}
    },
    "Rachel": {
        "profession": "Fashion (Ralph Lauren), formerly spoiled rich girl",
        "known_topics": ["fashion", "relationships", "shopping", "independence",
                         "career", "family", "New York social life"],
        "defense_mechanisms": ["charm", "deflection", "shopping"],
        "speech_patterns": ["Oh my God!", "Oh no no no no no"],
        "core_fear": "becoming her mother, losing independence",
        "personality_scores": {"vanity": 0.75, "growth": 0.9,
                                "charm": 0.9, "emotional_intelligence": 0.75}
    }
}

ENRICHMENT_PROMPT = """You are a dialogue annotation expert for the TV show Friends.

Given a dialogue line and its surrounding context, annotate it with:
- emotion: primary emotion behind the line (anxiety, deflection, warmth, humor, sarcasm, sadness, excitement, confusion, love, annoyance)
- intent: what the speaker is trying to do (deflect, comfort, tease, assert, question, admit, avoid, connect, reject, impress)
- defense_mechanism: if applicable (sarcasm, humor, intellectualizing, magical_thinking, charm, naivety, control, none)
- topic: main topic (romance, friendship, work, food, family, existential, trivial, crisis)
- responding_to: brief description of what triggered this line
- to_whom: who is being addressed

Respond ONLY with a JSON object, no other text.

Character profile: {profile}

Context window (lines before):
{context}

Line to annotate:
Speaker: {speaker}
Quote: {quote}

JSON:"""

def get_context_window(rows, idx, window=3):
    """Get N lines before current line as context."""
    start = max(0, idx - window)
    context_lines = []
    for i in range(start, idx):
        r = rows[i]
        context_lines.append(f"{r['author']}: {r['quote']}")
    return "\n".join(context_lines) if context_lines else "Start of scene"

def enrich_line(speaker, quote, context, profile):
    """Call Groq to enrich a single dialogue line."""
    prompt = ENRICHMENT_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        context=context,
        speaker=speaker,
        quote=quote
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        return {
            "emotion": "unknown",
            "intent": "unknown",
            "defense_mechanism": "none",
            "topic": "unknown",
            "responding_to": "unknown",
            "to_whom": "unknown",
            "error": str(e)
        }

def process_csv(input_path, output_path, target_characters=None):
    """Process CSV and output enriched JSONL."""
    if target_characters is None:
        target_characters = list(CHARACTER_PROFILES.keys())

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")
    processed = 0
    skipped = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for idx, row in enumerate(rows):
            speaker = row['author'].strip()

            # Skip group lines and non-target characters
            if ',' in speaker or speaker not in target_characters:
                skipped += 1
                continue

            profile = CHARACTER_PROFILES.get(speaker, {})
            context = get_context_window(rows, idx, window=3)
            quote = row['quote'].strip()

            # Skip very short lines
            if len(quote) < 5:
                skipped += 1
                continue

            annotations = enrich_line(speaker, quote, context, profile)

            record = {
                "speaker": speaker,
                "quote": quote,
                "context_window": context,
                "emotion": annotations.get("emotion", "unknown"),
                "intent": annotations.get("intent", "unknown"),
                "defense_mechanism": annotations.get("defense_mechanism", "none"),
                "topic": annotations.get("topic", "unknown"),
                "responding_to": annotations.get("responding_to", "unknown"),
                "to_whom": annotations.get("to_whom", "unknown"),
                "character_profile": {
                    "profession": profile.get("profession", ""),
                    "known_topics": profile.get("known_topics", []),
                    "personality_scores": profile.get("personality_scores", {})
                }
            }

            out.write(json.dumps(record) + "\n")
            processed += 1

            if processed % 10 == 0:
                print(f"Processed {processed} lines...")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "friends_dialogues.csv"
    output_jsonl = sys.argv[2] if len(sys.argv) > 2 else "enriched_dialogues.jsonl"
    process_csv(input_csv, output_jsonl)
