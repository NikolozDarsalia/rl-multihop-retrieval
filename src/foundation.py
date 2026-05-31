"""Step 1 - build and freeze the canonical corpus from MuSiQue-Ans.

Pools paragraphs across questions, dedupes them to canonical nodes, and records
each question's gold chain (in hop order) and its candidate pool. Writes
nodes.parquet, questions.parquet, and a manifest (with a hash that detects any
later node_id churn).

``build_corpus`` takes plain iterables of MuSiQue-shaped dicts, so it runs
identically on the real HF dataset and on the synthetic fixture in fixture.py.
"""

from __future__ import annotations
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from . import config, schema


def _norm(s: str | None) -> str:
    return (s or "").strip()


def build_corpus(
    records_by_split: dict[str, list],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """records_by_split: {"train": <iterable of records>, "validation": <...>}.

    Each record is a MuSiQue example with keys: id, question, answer,
    paragraphs[{idx, title, paragraph_text, is_supporting}],
    question_decomposition[{paragraph_support_idx, answer, ...}].
    """
    key_to_id: dict[tuple[str, str], int] = {}
    nodes: list[tuple[int, str, str]] = []  # (node_id, title, text)
    freq: Counter = Counter()
    question_rows: list[dict] = []
    skipped = 0

    def get_node_id(title: str, text: str) -> int:
        key = (_norm(title), _norm(text))
        nid = key_to_id.get(key)
        if nid is None:
            nid = len(nodes)
            key_to_id[key] = nid
            nodes.append((nid, key[0], key[1]))
        return nid

    for split, records in records_by_split.items():
        for ex in records:
            paras = {p["idx"]: p for p in ex["paragraphs"]}
            decomp = ex["question_decomposition"]
            gold_idx = [h.get("paragraph_support_idx") for h in decomp]
            if any(g is None or g not in paras for g in gold_idx):
                skipped += 1
                continue

            cand_ids = sorted(
                {get_node_id(p["title"], p["paragraph_text"]) for p in paras.values()}
            )
            for nid in cand_ids:
                freq[nid] += 1

            gold_ids = [
                get_node_id(paras[g]["title"], paras[g]["paragraph_text"])
                for g in gold_idx
            ]

            question_rows.append(
                dict(
                    question_id=ex["id"],
                    split=split,
                    query_text=ex["question"],
                    n_hops=len(decomp),
                    gold_node_ids=gold_ids,
                    candidate_node_ids=cand_ids,
                    answer=ex.get("answer", ""),
                )
            )

    nodes_df = pd.DataFrame(nodes, columns=["node_id", "title", "text"])
    nodes_df["freq"] = nodes_df["node_id"].map(freq).fillna(0).astype(int)
    nodes_df = nodes_df[schema.NODE_COLUMNS]

    questions_df = pd.DataFrame(question_rows, columns=schema.QUESTION_COLUMNS)

    manifest = dict(
        built_at=datetime.now(timezone.utc).isoformat(),
        dataset=config.DATASET_NAME,
        subset=config.DATASET_SUBSET,
        n_train_questions=config.N_TRAIN_QUESTIONS,
        n_val_questions=config.N_VAL_QUESTIONS,
        seed=config.SEED,
        n_nodes=int(len(nodes_df)),
        n_questions=int(len(questions_df)),
        n_skipped=int(skipped),
        node_id_hash=_freeze_hash(nodes_df),
    )
    return nodes_df, questions_df, manifest


def _freeze_hash(nodes_df: pd.DataFrame) -> str:
    """Stable hash of (title, text) in node_id order. If this changes between
    runs, your node_ids churned and every downstream artifact is invalidated."""
    h = hashlib.sha256()
    for title, text in zip(nodes_df["title"], nodes_df["text"]):
        h.update(title.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_musique_records() -> dict[str, list]:
    """Load the answerable subset from HF, then apply the configured slice + seed."""
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    out: dict[str, list] = {}
    for split, n in (
        ("train", config.N_TRAIN_QUESTIONS),
        ("validation", config.N_VAL_QUESTIONS),
    ):
        d = ds[split].shuffle(seed=config.SEED)
        if n is not None:
            d = d.select(range(min(n, len(d))))
        out[split] = d
    return out


def main() -> None:
    records = load_musique_records()
    nodes_df, questions_df, manifest = build_corpus(records)
    schema.save_nodes(nodes_df, config.NODES_PATH)
    schema.save_questions(questions_df, config.QUESTIONS_PATH, n_nodes=len(nodes_df))
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nnodes     -> {config.NODES_PATH}")
    print(f"questions -> {config.QUESTIONS_PATH}")
    print(f"manifest  -> {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
