"""The data contract for the foundation artifacts.

Two tables, keyed on a canonical integer ``node_id``:

nodes.parquet      one row per unique paragraph
    node_id   int    canonical id, 0..N-1 (FROZEN once built)
    title     str    Wikipedia article title
    text      str    paragraph text
    freq      int    number of questions whose candidate pool contains this node

questions.parquet  one row per question
    question_id          str
    split                str        "train" | "validation"
    query_text           str
    n_hops               int        2..4
    gold_node_ids        list[int]  canonical ids of the gold paragraphs, IN HOP ORDER
    candidate_node_ids   list[int]  canonical ids of all (<=20) paragraphs for the question
    answer               str        final answer (only needed for the optional QA eval)

Everything else (embeddings, edges, communities) is built later and keyed on
``node_id``.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

NODE_COLUMNS = ["node_id", "title", "text", "freq"]
QUESTION_COLUMNS = [
    "question_id",
    "split",
    "query_text",
    "n_hops",
    "gold_node_ids",
    "candidate_node_ids",
    "answer",
]


def validate_nodes(df: pd.DataFrame) -> None:
    assert list(df.columns) == NODE_COLUMNS, f"node columns are {list(df.columns)}"
    assert df["node_id"].is_unique, "node_id must be unique"
    assert (df["node_id"].to_numpy() == np.arange(len(df))).all(), (
        "node_id must be 0..N-1, contiguous"
    )
    assert df["title"].notna().all() and df["text"].notna().all(), (
        "title/text must be non-null"
    )


def validate_questions(df: pd.DataFrame, n_nodes: int) -> None:
    assert list(df.columns) == QUESTION_COLUMNS, (
        f"question columns are {list(df.columns)}"
    )
    assert df["question_id"].is_unique, "question_id must be unique"
    assert df["n_hops"].between(2, 4).all(), "n_hops must be in 2..4"
    for gold, cand, n in zip(
        df["gold_node_ids"], df["candidate_node_ids"], df["n_hops"]
    ):
        assert len(gold) == n, "len(gold_node_ids) must equal n_hops"
        assert all(0 <= g < n_nodes for g in gold), "gold node id out of range"
        assert set(gold).issubset(set(cand)), (
            "gold ids must be a subset of candidate ids"
        )


def save_nodes(df: pd.DataFrame, path) -> None:
    validate_nodes(df)
    df.to_parquet(path, index=False)


def load_nodes(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_nodes(df)
    return df


def save_questions(df: pd.DataFrame, path, n_nodes: int) -> None:
    validate_questions(df, n_nodes)
    df.to_parquet(path, index=False)


def load_questions(path, n_nodes: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_questions(df, n_nodes)
    return df
