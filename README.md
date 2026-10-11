# RL for Multi-Hop Retrieval (MuSiQue)

Vanilla vector RAG ranks documents by query–document cosine similarity. On multi-hop questions this misses the later-hop evidence: in our EDA about 50% of the hop ≥ 2 gold paragraphs fall outside the top-5 candidates. This project tests whether a small reinforcement-learning policy can improve retrieval over frozen BGE embeddings, and compares it with supervised learning-to-rank and a cross-encoder.

- **Dataset:** [MuSiQue-Ans](https://huggingface.co/datasets/bdsaglam/musique) (`answerable` subset). It has 22,355 questions (19,938 train, 2,417 validation) and 101,962 unique paragraphs after deduplication.
- **Encoder:** `BAAI/bge-base-en-v1.5` (768-d, L2-normalised).
- **Final approaches:**
  1. A *contextual-bandit reranker* (RL, self-critical REINFORCE) that reorders a candidate pool.
  2. A *sequential multi-hop policy* (RL) that picks documents one at a time and updates a context vector after each pick.
- **Clustering:** we built a hierarchical K-Means over the corpus to set up a cluster-navigating RL agent, but we did **not** use it in the final approach. It remains as extra data insight and a starting point for future work (see [Clustering](#clustering-explored-not-used-in-the-final-pipeline)).

## Repository layout

```
├── src/                      Reusable Python modules (see "Code reference")
├── notebooks/                Numbered notebooks, run in order (see "Notebooks")
│   └── approach.md           Design doc for the original cluster-navigation MDP (not used in the end)
├── data/                     Git-ignored, rebuilt by the steps below
│   ├── raw/                  HF dataset cache
│   ├── interim/              nodes.parquet, questions.parquet, corpus_manifest.json
│   └── artifacts/            doc_emb.npy, query_emb.npy, query_ids.npy, reranker_policy.pt,
│                             clustering_output/ (clustering artifacts)
├── checkpoints/              Written by src/rl.py (empty by default)
├── tokens/token_hf.txt       Optional HuggingFace token (git-ignored)
├── pyproject.toml, uv.lock   Dependencies (Python 3.13)
└── main.py                   Placeholder, not used
```

## Setup

Requirements: Python 3.13 (`.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd rl-graph-rag-retrieval
uv sync                      # creates .venv from uv.lock
uv pip install peft          # only for notebook 06 fine-tuning (not in pyproject)
uv pip install FlagEmbedding # only for src/rl.py, the unused cluster-RL script
source .venv/bin/activate    # then: jupyter lab   (or pick the .venv kernel in VS Code)
```

Notes:
- Always run notebooks from `notebooks/`. They use relative paths like `../data/...` and `sys.path.insert(0, "..")`.
- **Optional HF token:** `01_EDA.ipynb` logs in with the token in `tokens/token_hf.txt`. The dataset is public, so you can skip the login cell.
- **macOS:** torch, xgboost and lightgbm each ship an OpenMP runtime, which can crash the kernel. The first cell of notebook 05 sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`. Keep those lines.
- **Hardware:** everything runs on CPU/MPS except embedding creation (03) and BGE fine-tuning (06), which need a GPU (we used Colab and Kaggle).

## Reproducing the experiments

Run the steps in order. Each one produces the files the next one reads.

| # | Step | Command / notebook | Produces |
|---|------|--------------------|----------|
| 1 | Build corpus | `python -m src.foundation` (or `02_build_corpus.ipynb`) | `data/interim/nodes.parquet`, `questions.parquet`, `corpus_manifest.json` |
| 2 | Embeddings (GPU) | `python -m src.embeddings`, or the Colab cell in `03_create_embeddings.ipynb` | `data/artifacts/doc_emb.npy`, `query_emb.npy`, `query_ids.npy` |
| 3 | Closed-pool reranker + baselines + open-corpus sweep | `05_contextual_bandit_reranker.ipynb` | Metrics, plots, `data/artifacts/reranker_policy.pt` |
| 4 | Sequential policy (GPU) | `06_sequential_decision_making.ipynb` | Fine-tuned LoRA adapter, new embeddings, MHP results |
| – | *Optional:* clustering | `04_clustering.ipynb` | `data/artifacts/clustering_output/` |

Details and gotchas:
- **Step 1** downloads MuSiQue from HuggingFace (needs internet). The seed is fixed (`199806` in `src/config.py`) and `corpus_manifest.json` stores a hash of the node table so you can detect any `node_id` drift.
- **Step 2:** the artifacts we used were created on a Colab GPU. That run embedded `title + " - " + text` (see the commented cell in notebook 03). `src/embeddings.py` encodes `text` only, so to match our numbers exactly use the notebook's formulation. Queries get the BGE instruction prefix. A full run encodes about 102k paragraphs and about 22k queries. Row `i` of `doc_emb` is `node_id == i`.
- **Step 3** is the main result. Runtime on CPU is a few minutes for the bandit and tree models, plus longer for the cross-encoder comparison and the top-N sweep. Set `INCLUDE_CE = False` to skip the cross-encoder in the per-hop cell.
- **Step 4** was developed on Kaggle. The paths at the top of notebook 06 (`ARTIFACTS_DIR`, `INTERIM_DIR`, `NEW_*_EMB_PATH`, `BEST_CHECKPOINT_PATH`) point to `/kaggle/...`. Edit them to your local `data/` folders to run elsewhere. Fine-tuning needs CUDA, and the notebook contains all of its own code (nothing from `src/`).

## Notebooks

| Notebook | What it shows |
|----------|---------------|
| `01_EDA.ipynb` | MuSiQue structure (2/3/4-hop, 20 paragraphs per question, 41% duplicate paragraphs across questions). Shows the multi-hop problem: query similarity of gold paragraphs drops after hop 1, about 50% of later-hop golds fall outside the top-5, and the previous hop's bridge entity appears in the next gold paragraph about 80% of the time. |
| `02_build_corpus.ipynb` | Smoke test of `build_corpus` on the synthetic fixture. Inspects `nodes` and `questions` tables, hop and split counts, and a few sample records. Includes corpus EDA (split balance, TF-IDF similarity, MiniLM multi-hop diagnostics). |
| `03_create_embeddings.ipynb` | The (commented-out) Colab GPU embedding code. Sanity checks on embeddings: gold vs distractor similarity, rank by hop. Vector-RAG baseline Recall@K up to K=100 (first-stage ceiling about 65.5% at K=50 and 71.4% at K=100). Two-stage baseline with the `bge-reranker-base` cross-encoder and its latency. |
| `04_clustering.ipynb` | **Exploratory, not used downstream.** Builds the two-level K-Means hierarchy, validates cluster quality, and analyses how many clusters a query's gold spans. Also runs an "oracle cluster routing" experiment and exports precomputed tables. |
| `05_training_rl.ipynb` | **Exploratory, not used downstream.** Runs the cluster-navigation RL script `src/rl.py` and checks how highly the fine clusters containing the gold paragraphs rank for a query. |
| `05_contextual_bandit_reranker.ipynb` | **Main experiment.** Bandit reranker vs cosine, XGBoost/LightGBM rankers and a cross-encoder on the ≤20-candidate pool. Also covers by-hop analysis, latency, a feature ablation, the open-corpus top-N sweep (N = 50/75/100/200), and saving the policy. |
| `06_sequential_decision_making.ipynb` | Sequential multi-hop policy over a global top-100 pool. Diagnoses why it plateaus on frozen BGE (the "signal gap"), LoRA-fine-tunes BGE on hop chains, then trains a dual-embedding policy and evaluates it, including a per-hop breakdown. |

### Headline results (validation, n = 2,417)

Closed pool (≤20 candidates, gold always present), in `05_contextual_bandit_reranker.ipynb`:

| Model | R@5 | MAP |
|-------|-----|-----|
| Cosine baseline | 0.703 | 0.671 |
| Cross-encoder (`bge-reranker-base`) | 0.712 | 0.689 |
| RL contextual bandit | 0.765 | 0.721 |
| XGBRanker / LGBMRanker | 0.773 / 0.772 | 0.729 / 0.728 |

- The gain comes entirely from the inter-document ("bridge") feature `s_prf`. In the ablation, the query-only ranker equals cosine (MAP 0.6705).
- RL roughly matches supervised LambdaMART and is much faster than the cross-encoder.
- In the open-corpus setting, reranking is capped by first-stage recall (ceiling 0.66 at N=50, 0.76 at N=200).

Sequential policy (global top-100 pool), in `06_sequential_decision_making.ipynb`:
- Overall MAP goes from 0.404 (vector RAG) to 0.434.
- The gain is concentrated on harder questions: MAP +0.053 for 3-hop and +0.068 for 4-hop, but only +0.004 for 2-hop.

## Code reference (`src/`)

**`config.py`**: paths (`ROOT`, `DATA_DIR`, `INTERIM_DIR`, `ARTIFACTS_DIR`, ...), dataset name and subset, question limits (`N_TRAIN_QUESTIONS`/`N_VAL_QUESTIONS`, `None` = full), `SEED`.

**`schema.py`**: data contract for the two tables (`nodes`: `node_id,title,text,freq`; `questions`: `question_id,split,query_text,n_hops,gold_node_ids,candidate_node_ids,answer`).
- `validate_nodes`, `validate_questions`: assert the contract.
- `save_nodes`, `load_nodes`, `save_questions`, `load_questions`: parquet I/O with validation.

**`foundation.py`**: build the canonical corpus. Run with `python -m src.foundation`.
- `load_musique_records()`: load, shuffle and slice MuSiQue-Ans from HuggingFace.
- `build_corpus(records_by_split)`: dedupe paragraphs into `node_id`s and record each question's gold chain (hop order) and candidate pool. Returns `(nodes, questions, manifest)`.
- `_freeze_hash(nodes_df)`: SHA-256 of the node table, used to detect `node_id` drift.
- `main()`: build and write the interim files.

**`fixture.py`**: `make_fixture(n_questions, n_paragraphs, seed)` builds a tiny synthetic MuSiQue-shaped dataset for offline tests.

**`embeddings.py`**: frozen BGE encoding. Run with `python -m src.embeddings`.
- `encode_texts(model, texts, ...)`: batched, L2-normalised encoding.
- `build_embeddings(docs_df, queries_df, ...)`: encode docs and queries (BGE instruction prefix on queries) and create a manifest.
- `save_embeddings`, `load_embeddings`: persist or load `.npy` files plus manifest.

**`contextual_bandit_reranker.py`**: main reranking module (imported as `cbr`).
- Data: `load_artifacts()` and `build_artifact_dict(...)` build the shared dict `a`. `iter_query_pools(a, split)` yields a `QueryPool` per question.
- Features: `build_features(query_vec, cand_vecs, ...)` returns z-scored `s_query, s_prf, s_centroid, s_maxother, rank_q`. `feature_names` lists them.
- Metrics: `recall_at_k`, `mrr`, `average_precision`.
- Policy: `RerankPolicy` (small MLP scorer), `sample_plackett_luce` (sample a ranking), `greedy_ranking`.
- Training and evaluation: `TrainConfig`, `train_reranker` (self-critical REINFORCE, optionally with ablation via `drop_features`), `evaluate` (policy vs cosine, optionally by hop).
- Persistence: `save_policy`, `load_policy`.
- Baselines: `train_tree_ranker` / `evaluate_tree_ranker` (XGBoost and LightGBM LambdaMART), `evaluate_cross_encoder` (HF cross-encoder).

**`topk_retrieval_experiment.py`**: open-corpus extension (imported as `tk`).
- `build_topn_pools(a, split, n, inject_gold)`: candidates from global top-N cosine retrieval. Gold is injected for training only.
- `first_stage_recall`: the recall ceiling for a given N.
- `recall_at_k`, `mrr`, `average_precision`: metrics with the true gold count as denominator.
- `ResidualPolicy`: cosine score plus a zero-initialised MLP correction, so training starts from the baseline.
- `train_rl` (truncated Plackett-Luce REINFORCE), `train_tree`.
- `score_model`: score a trained model on eval pools.
- `run_topn_sweep(a, ns, ks, models, ...)`: full N sweep, returning a DataFrame.
- `pivot_metric`: models × N summary.

**`clustering.py`** (used only by notebook 04): see below.

**`rl.py`** (used only by notebook 05_training_rl, **not part of the final pipeline**): REINFORCE actor-critic over the cluster hierarchy.
- `load_artifacts`, `extract_top_k`, `rerank`, `Episode` (environment).
- `ActorCritic` (model), `run_episode`, `collect_batch`, `update`, `evaluate`, `train`.
- Run with `%run ../src/rl.py --device cpu` from `notebooks/`. It needs `clustering_output/` and `FlagEmbedding`, and hard-codes relative `..` paths.

The sequential-policy code (`precompute_topk_cosines`, `MHPPolicy`, `ContextUpdater`, `train_mhp_learned_dual`, `evaluate_mhp_learned_dual`, `finetune_bge`, `recompute_embeddings`, ...) lives **inside notebook 06**, not in `src/`.

## Clustering (explored, not used in the final pipeline)

The original idea was an RL agent that navigates a two-level cluster hierarchy of the corpus (`notebooks/approach.md`). We implemented the clustering and the environment, but dropped the approach. Instead, the clustering became a data-analysis tool and may be useful for future work.

What exists:
- `src/clustering.py`: `get_average_random_pair_similarity` (random-pair baseline), `check_coarse_elbow` / `plot_elbow_curve` (choose coarse K), `get_cluster_metrics_table` and `format_hierarchical_metrics` (intra vs inter-cluster similarity and separation), `find_best_fine_kmeans`, `build_dynamic_hierarchy` (coarse K-Means with K=20, then a fine K per coarse cluster, giving 169 fine clusters), `apply_hierarchy_to_queries`, `plot_hierarchy_tree`, `save_artifacts`.
- Outputs in `data/artifacts/clustering_output/`: cluster labels for documents and queries, fitted K-Means models, similarity tables, and precomputed RL tables.
- Insights from notebooks 04 and 05: gold paragraphs of a query typically span about 2.15 fine clusters. Oracle routing into the gold clusters gives about 77% recall with about 54 retrieved docs on average, vs about 74% for flat top-100 vector RAG. A gold paragraph's fine cluster is the query's nearest centroid 32% of the time (85% within the 20 nearest).

None of this feeds into notebooks 05 (bandit reranker) or 06 (sequential policy), which use only the embeddings.
