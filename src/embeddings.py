"""Encode documents and queries with a frozen retrieval encoder.

No training happens here - the encoder is pretrained and fixed. We just run
forward passes and persist the vectors.

Storage layout (in data/artifacts/):
    doc_emb.npy      float32 [n_docs, d]      row index == node_id (0..N-1)
    query_emb.npy    float32 [n_queries, d]   row i aligned to query_ids[i]
    query_ids.npy    str     [n_queries]      question_id for each query row
    embeddings_manifest.json                  encoder, dim, counts, corpus hash

The manifest stores the corpus node_id_hash so a later step can refuse to use
embeddings that were built against a different (churned) node table.
"""

from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config
from .foundation import _freeze_hash

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
# BGE wants this instruction on QUERIES only; passages are encoded as-is.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DOC_EMB_PATH = config.ARTIFACTS_DIR / "doc_emb.npy"
QUERY_EMB_PATH = config.ARTIFACTS_DIR / "query_emb.npy"
QUERY_IDS_PATH = config.ARTIFACTS_DIR / "query_ids.npy"
EMB_MANIFEST_PATH = config.ARTIFACTS_DIR / "embeddings_manifest.json"


def encode_texts(
    model, texts: list[str], batch_size: int = 256, instruction: str | None = None
) -> np.ndarray:
    """Encode a list of texts to L2-normalized float32 vectors."""
    if instruction:
        texts = [instruction + t for t in texts]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine == dot product downstream
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def build_embeddings(
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    query_instruction: str | None = BGE_QUERY_INSTRUCTION,
    batch_size: int = 256,
    device: str | None = None,
):
    """Encode all docs and queries. Returns (doc_emb, query_emb, query_ids, manifest)."""
    # The row==node_id contract: docs must be in contiguous node_id order.
    assert (docs_df["node_id"].to_numpy() == np.arange(len(docs_df))).all(), (
        "docs_df must be sorted by node_id (0..N-1) so row index == node_id"
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    doc_emb = encode_texts(model, docs_df["text"].tolist(), batch_size=batch_size)
    query_emb = encode_texts(
        model,
        queries_df["query_text"].tolist(),
        batch_size=batch_size,
        instruction=query_instruction,
    )
    query_ids = queries_df["question_id"].to_numpy().astype(str)

    manifest = dict(
        model_name=model_name,
        dim=int(doc_emb.shape[1]),
        n_docs=int(doc_emb.shape[0]),
        n_queries=int(query_emb.shape[0]),
        normalized=True,
        query_instruction=query_instruction or "",
        corpus_node_id_hash=_freeze_hash(
            docs_df
        ),  # binds embeddings to this exact corpus
    )
    return doc_emb, query_emb, query_ids, manifest


def save_embeddings(doc_emb, query_emb, query_ids, manifest) -> None:
    np.save(DOC_EMB_PATH, doc_emb)
    np.save(QUERY_EMB_PATH, query_emb)
    np.save(QUERY_IDS_PATH, np.asarray(query_ids).astype(str))
    EMB_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load_embeddings():
    """Returns (doc_emb, query_emb, query_ids, manifest)."""
    doc_emb = np.load(DOC_EMB_PATH)
    query_emb = np.load(QUERY_EMB_PATH)
    query_ids = np.load(QUERY_IDS_PATH, allow_pickle=False)
    manifest = json.loads(EMB_MANIFEST_PATH.read_text())
    return doc_emb, query_emb, query_ids, manifest


def main() -> None:
    from . import schema

    docs_df = schema.load_nodes(config.NODES_PATH)
    queries_df = pd.read_parquet(config.QUESTIONS_PATH)
    doc_emb, query_emb, query_ids, manifest = build_embeddings(docs_df, queries_df)
    save_embeddings(doc_emb, query_emb, query_ids, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\ndoc_emb   {doc_emb.shape}  -> {DOC_EMB_PATH}")
    print(f"query_emb {query_emb.shape}  -> {QUERY_EMB_PATH}")


if __name__ == "__main__":
    main()
