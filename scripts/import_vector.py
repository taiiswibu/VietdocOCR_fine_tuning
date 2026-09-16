"""Optional local vector DB ingestion; explicit model download on first execution."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("jsonl",type=Path);p.add_argument("--db",default="workspace/chroma");p.add_argument("--collection",default="vietdoc")
    p.add_argument("--model",default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--query");a=p.parse_args()
    import chromadb
    from sentence_transformers import SentenceTransformer
    rows=[json.loads(l) for l in a.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:raise ValueError("Không có chunks")
    model=SentenceTransformer(a.model,device="cpu")
    client=chromadb.PersistentClient(path=a.db)
    collection=client.get_or_create_collection(a.collection,metadata={"embedding_model":a.model})
    if (collection.metadata or {}).get("embedding_model")!=a.model:raise ValueError("Collection đã dùng embedding model khác. Chọn collection mới.")
    # Replace this document's old chunks, so shorter corrected documents leave no stale vectors.
    for document_id in {r['metadata']['document_id'] for r in rows}:collection.delete(where={"document_id":document_id})
    for i in range(0,len(rows),32):
        batch=rows[i:i+32];vectors=model.encode([r["text"] for r in batch],normalize_embeddings=True).tolist()
        collection.upsert(ids=[r["id"] for r in batch],documents=[r["text"] for r in batch],metadatas=[r["metadata"] for r in batch],embeddings=vectors)
    print(f"Imported {len(rows)} chunks into {a.collection}")
    if a.query:
        result=collection.query(query_embeddings=model.encode([a.query],normalize_embeddings=True).tolist(),n_results=min(5,collection.count()))
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
