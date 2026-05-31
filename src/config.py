"""Central configuration: paths and build parameters.

Everything downstream imports from here, so paths and knobs live in one place.
"""

from __future__ import annotations
from pathlib import Path

# Repo root = the parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # untouched downloads / HF cache
INTERIM_DIR = DATA_DIR / "interim"  # foundation outputs (node + question tables)
ARTIFACTS_DIR = DATA_DIR / "artifacts"  # graph outputs (embeddings, edges, communities)

for _d in (DATA_DIR, RAW_DIR, INTERIM_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
DATASET_NAME = "bdsaglam/musique"
DATASET_SUBSET = (
    "answerable"  # MuSiQue-Ans. Do NOT use "default" (mixes in unanswerable).
)

# --- Corpus scope (the Step 0 decision: bounded global pooled corpus) ---
# Start on the slice you profiled in the EDA, then raise these once the
# pipeline runs end to end. Set to None to use the full split.
N_TRAIN_QUESTIONS = None
N_VAL_QUESTIONS = None

SEED = 199806

# --- Foundation artifact paths ---
NODES_PATH = INTERIM_DIR / "nodes.parquet"
QUESTIONS_PATH = INTERIM_DIR / "questions.parquet"
MANIFEST_PATH = INTERIM_DIR / "corpus_manifest.json"
