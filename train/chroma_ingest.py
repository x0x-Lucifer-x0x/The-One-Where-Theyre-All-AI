"""
Step 3: Ingest ChromaDB documents per character.
Creates 6 separate collections — one per character.
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, local, fast

def ingest_character(char_name, docs_path, client, model):
    if not os.path.exists(docs_path):
        print(f"Skipping {char_name} — {docs_path} not found")
        return

    collection_name = f"persona_{char_name.lower()}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"character": char_name}
    )

    with open(docs_path, 'r') as f:
        docs = [json.loads(line) for line in f if line.strip()]

    if not docs:
        print(f"No docs for {char_name}")
        return

    ids        = [d["id"]       for d in docs]
    documents  = [d["document"] for d in docs]
    metadatas  = [d["metadata"] for d in docs]

    # Batch embed locally
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=documents,
                   metadatas=metadatas, embeddings=embeddings)
    print(f"[ChromaDB] {char_name}: {len(docs)} docs ingested into '{collection_name}'")

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    model  = SentenceTransformer(EMBEDDING_MODEL)

    characters = ["Chandler", "Joey", "Monica", "Ross", "Phoebe", "Rachel"]
    for char in characters:
        ingest_character(char, f"chroma_docs_{char.lower()}.jsonl", client, model)

    print("\nAll collections ready. Test with:")
    print("  python query_persona.py Chandler 'do you like someone?'")

if __name__ == "__main__":
    main()
