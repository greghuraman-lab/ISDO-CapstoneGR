"""
ISDO Knowledge Base Indexer
----------------------------
1. Reads all .md files from data/kb/
2. Splits each file into chunks at "## " (H2) headings
3. Stores the chunks in a persistent ChromaDB collection called 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence
   score for each

Dependencies: chromadb only (`pip install chromadb`)
Run from the project root (the folder that contains data/kb/):
    python build_kb_index.py
"""

import os
import re
import glob

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KB_DIR = os.path.join("data", "kb")
CHROMA_DB_DIR = "chroma_store"          # persisted on disk so the index survives reruns
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "My VPN keeps disconnecting right after I changed my password",
    "I forgot my password and my account is locked",
    "SAP login is failing with a DBCON_FAIL error for several users",
    "Outlook on my phone is not syncing new emails",
]


# ---------------------------------------------------------------------------
# Step 1 & 2: read markdown files and split into chunks at "## " headings
# ---------------------------------------------------------------------------
def load_and_chunk_markdown(kb_dir):
    """
    Returns a list of dicts:
        {
            "id": unique chunk id,
            "text": chunk text (title + section content),
            "source_file": filename,
            "article_title": the H1 title of the article,
            "section": the H2 heading this chunk belongs to (or "Overview"),
        }
    """
    chunks = []
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))

    if not md_files:
        raise FileNotFoundError(f"No .md files found in '{kb_dir}'")

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Grab the H1 title (first line starting with "# ") for context
        h1_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        article_title = h1_match.group(1).strip() if h1_match else filename

        # Split on H2 headings ("## Something"), keeping the heading text.
        # Everything before the first "## " becomes an "Overview" chunk.
        parts = re.split(r"(?m)^##\s+(.+)$", content)
        # parts[0] = text before first H2 heading
        # parts[1] = heading 1, parts[2] = body 1, parts[3] = heading 2, ...

        overview_text = parts[0].strip()
        if overview_text:
            chunks.append({
                "id": f"{filename}::overview",
                "text": f"{article_title}\n\n{overview_text}",
                "source_file": filename,
                "article_title": article_title,
                "section": "Overview",
            })

        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            chunk_text = f"{article_title} - {heading}\n\n{body}"
            chunks.append({
                "id": f"{filename}::{heading}".replace(" ", "_"),
                "text": chunk_text,
                "source_file": filename,
                "article_title": article_title,
                "section": heading,
            })

    return chunks


# ---------------------------------------------------------------------------
# Step 3: store chunks in ChromaDB
# ---------------------------------------------------------------------------
def build_collection(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Start fresh each run so re-running the script doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance -> easy confidence score
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source_file": c["source_file"],
                "article_title": c["article_title"],
                "section": c["section"],
            }
            for c in chunks
        ],
    )
    return collection


# ---------------------------------------------------------------------------
# Step 4: run sample queries and report the best-matching article
# ---------------------------------------------------------------------------
def run_sample_queries(collection, queries):
    print("\n" + "=" * 70)
    print("SAMPLE QUERY RESULTS")
    print("=" * 70)

    for query in queries:
        results = collection.query(query_texts=[query], n_results=1)

        best_metadata = results["metadatas"][0][0]
        best_distance = results["distances"][0][0]
        best_section = results["documents"][0][0].splitlines()[0]

        # Cosine distance is 0 (identical) .. 2 (opposite); convert to a
        # 0-1-ish confidence score where higher = more relevant.
        confidence = 1 - best_distance

        print(f"\nQuery: \"{query}\"")
        print(f"  Best matching article : {best_metadata['source_file']}")
        print(f"  Article title         : {best_metadata['article_title']}")
        print(f"  Matched section       : {best_metadata['section']}")
        print(f"  Confidence score      : {confidence:.4f}  (cosine distance = {best_distance:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading markdown files from '{KB_DIR}' ...")
    chunks = load_and_chunk_markdown(KB_DIR)
    print(f"Loaded {len(chunks)} chunks from "
          f"{len(set(c['source_file'] for c in chunks))} articles.")

    print(f"\nBuilding ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_DB_DIR}' ...")
    collection = build_collection(chunks)
    print(f"Stored {collection.count()} chunks in the collection.")

    run_sample_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()
