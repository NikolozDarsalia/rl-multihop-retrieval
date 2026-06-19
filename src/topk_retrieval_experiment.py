"""Open-corpus top-N retrieval sweep: how do the rerankers behave when the
candidate set is built by global cosine retrieval (top-N over the WHOLE corpus)
instead of the fixed <=20 per-question pool?

This is a different, harder task than the closed-pool reranker:
  * The gold is NOT guaranteed to be in the candidate set. Recall is therefore
    divided by the TRUE number of gold paragraphs (n_hops), so gold that the
    first stage failed to retrieve correctly counts as a miss.
  * Two forces fight as N grows: first-stage recall@N (the ceiling) rises, but
    the reranking problem gets harder (more distractors). We report the ceiling
    next to every model so the deltas are interpretable.

We reuse the building blocks from contextual_bandit_reranker (features, policy,
tree training, the BGE artifacts) and add: global retrieval, honest gold-count
metrics, and a truncated-Plackett-Luce REINFORCE loop that stays cheap at N=200.

No LLM / KG / agent / clusters. (Cross-encoder is intentionally omitted from the
sweep - scoring 200 text pairs per query is far too slow to run across 4 values
of N; add it back for a single small N if you want it in the table.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import contextual_bandit_reranker as cbr


# ==========================================================================
# 1. Pools built by global top-N cosine retrieval
# ==========================================================================
@dataclass
class TopNPool:
    question_id: str
    n_hops: int
    query_vec: np.ndarray
    cand_vecs: np.ndarray  # [n_cand, d], n_cand ~ N (+gold if injected)
    cand_node_ids: np.ndarray
    gold_local: np.ndarray  # positions in cand that ARE gold (subset)
    n_gold_total: int  # == n_hops; the recall denominator


def _topn_indices(q_block: np.ndarray, doc_emb: np.ndarray, n: int) -> np.ndarray:
    """Top-n doc rows by cosine for each query row in q_block. Returns [B, n]
    of doc indices, sorted by descending similarity. Embeddings are L2-normed,
    so dot product == cosine."""
    sims = q_block @ doc_emb.T  # [B, n_docs]
    n = min(n, doc_emb.shape[0])
    part = np.argpartition(-sims, kth=n - 1, axis=1)[:, :n]  # unsorted top-n
    # sort the n survivors by sim, descending
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return part[rows, order]


def build_topn_pools(
    a: dict, split: str, n: int, inject_gold: bool, query_chunk: int = 512
) -> list[TopNPool]:
    """Build one TopNPool per question via global top-N cosine retrieval.

    inject_gold=True (use for TRAIN): union the retrieved top-N with the gold
    nodes so the ranker always has positive labels to learn from.
    inject_gold=False (use for EVAL): strictly the retrieved top-N - this is the
    honest, real-world candidate set.
    """
    doc_emb, query_emb = a["doc_emb"], a["query_emb"]
    q2i = a["query_id_to_idx"]
    q = a["questions"]
    q = q[q["split"] == split] if split is not None else q

    rows = list(q.itertuples(index=False))
    qids = [str(r.question_id) for r in rows]
    valid = [(i, r) for i, r in enumerate(rows) if qids[i] in q2i]
    q_idx = np.array([q2i[qids[i]] for i, _ in valid])
    Q = query_emb[q_idx]

    # retrieve top-N in query chunks (avoids materializing a huge sim matrix)
    topn = np.empty((len(Q), min(n, doc_emb.shape[0])), dtype=np.int64)
    for s in range(0, len(Q), query_chunk):
        e = min(s + query_chunk, len(Q))
        topn[s:e] = _topn_indices(Q[s:e], doc_emb, n)

    pools = []
    for row_i, (orig_i, r) in enumerate(valid):
        gold_ids = [int(g) for g in r.gold_node_ids]  # node_id == doc row
        cand = list(topn[row_i])
        if inject_gold:
            missing = [g for g in gold_ids if g not in set(cand)]
            cand = cand + missing  # append unseen gold
        cand = np.asarray(cand, dtype=np.int64)
        cand_vecs = doc_emb[cand]
        pos = {int(c): j for j, c in enumerate(cand)}
        gold_local = np.array([pos[g] for g in gold_ids if g in pos], dtype=np.int64)
        pools.append(
            TopNPool(
                question_id=str(r.question_id),
                n_hops=int(r.n_hops),
                query_vec=query_emb[q_idx[row_i]],
                cand_vecs=cand_vecs,
                cand_node_ids=cand,
                gold_local=gold_local,
                n_gold_total=len(gold_ids),
            )
        )
    return pools


def first_stage_recall(pools: list[TopNPool]) -> float:
    """Mean fraction of gold captured by the first-stage top-N (the ceiling)."""
    return float(np.mean([len(p.gold_local) / p.n_gold_total for p in pools]))


# ==========================================================================
# 2. Honest metrics (denominator = true gold count, not just gold present)
# ==========================================================================
def recall_at_k(
    ranking: np.ndarray, gold_local: set, k: int, n_gold_total: int
) -> float:
    if n_gold_total == 0:
        return 0.0
    return len(set(ranking[:k].tolist()) & gold_local) / n_gold_total


def mrr(ranking: np.ndarray, gold_local: set) -> float:
    for r, idx in enumerate(ranking, start=1):
        if int(idx) in gold_local:
            return 1.0 / r
    return 0.0


def average_precision(ranking: np.ndarray, gold_local: set, n_gold_total: int) -> float:
    if n_gold_total == 0:
        return 0.0
    hits, score = 0, 0.0
    for r, idx in enumerate(ranking, start=1):
        if int(idx) in gold_local:
            hits += 1
            score += hits / r
    return score / n_gold_total  # missed gold penalizes via denominator


# ==========================================================================
# 3. Feature cache
# ==========================================================================
def _cache_features(pools: list[TopNPool]):
    out = []
    for p in pools:
        F = cbr.build_features(p.query_vec, p.cand_vecs)
        out.append((torch.from_numpy(F), set(p.gold_local.tolist()), p))
    return out


# ==========================================================================
# 4. Truncated-Plackett-Luce REINFORCE (cheap at large N)
# ==========================================================================
def _sample_topm(scores: torch.Tensor, m: int, temperature: float = 1.0):
    """Sample the first m positions without replacement. Returns (idx list, logp)."""
    s = scores / temperature
    remaining = list(range(len(s)))
    chosen, logp = [], scores.new_zeros(())
    for _ in range(min(m, len(s))):
        sub = s[remaining]
        probs = torch.softmax(sub, dim=0)
        c = torch.multinomial(probs, 1).item()
        logp = logp + torch.log(probs[c] + 1e-12)
        chosen.append(remaining.pop(c))
    return np.array(chosen, dtype=np.int64), logp


class ResidualPolicy(torch.nn.Module):
    """score = scale * cosine(feat col 0) + MLP correction.

    The MLP's last layer is zero-initialized, so at init the score is just a
    monotone function of cosine -> the policy STARTS at the cosine baseline and
    REINFORCE can only learn a correction on top of it. This is the warm-start
    that makes RL stable in the sparse-reward open-corpus regime.
    """

    def __init__(self, n_feat: int, hidden: int = 32, cos_col: int = 0):
        super().__init__()
        self.mlp = cbr.RerankPolicy(n_feat, hidden=hidden)
        last = self.mlp.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.cos_col = cos_col
        self.scale = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.scale * feats[:, self.cos_col] + self.mlp(feats)


def train_rl(
    pools_cache,
    n_feat: int,
    epochs: int = 10,
    lr: float = 2e-3,
    hidden: int = 32,
    top_m: int = 30,
    batch_size: int = 32,
    seed: int = 199806,
    residual: bool = True,
    verbose: bool = False,
):
    """Self-critical REINFORCE with MAP reward, truncated to the top-m positions.

    residual=True wraps the policy as a zero-initialized correction on cosine, so
    training starts at the baseline instead of a random ranking (strongly
    recommended for open-corpus top-N, where reward is sparse).
    """
    torch.manual_seed(seed)
    policy = (
        ResidualPolicy(n_feat, hidden=hidden)
        if residual
        else cbr.RerankPolicy(n_feat, hidden=hidden)
    )
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        order = rng.permutation(len(pools_cache))
        opt.zero_grad()
        loss_accum, in_batch = torch.zeros(()), 0
        for step, i in enumerate(order, start=1):
            feats, gold_set, p = pools_cache[i]
            if not gold_set:  # no positive -> no signal
                continue
            scores = policy(feats)
            samp, logp = _sample_topm(scores, top_m)
            greedy = cbr.greedy_ranking(scores.detach())[:top_m]
            adv = average_precision(samp, gold_set, p.n_gold_total) - average_precision(
                greedy, gold_set, p.n_gold_total
            )
            loss_accum = loss_accum + (-adv * logp)
            in_batch += 1
            if in_batch == batch_size or step == len(order):
                (loss_accum / max(in_batch, 1)).backward()
                opt.step()
                opt.zero_grad()
                loss_accum, in_batch = torch.zeros(()), 0
        if verbose:
            print(f"    rl epoch {epoch}")
    return policy


# ==========================================================================
# 5. Tree training adapter (reuses your feature matrix; honest gold labels)
# ==========================================================================
def _flatten_for_trees(pools_cache):
    X, y, gid, sizes = [], [], [], []
    for g, (feats, gold_set, p) in enumerate(pools_cache):
        F = feats.numpy()
        X.append(F)
        lab = np.zeros(len(F), dtype=int)
        if gold_set:
            lab[list(gold_set)] = 1
        y.append(lab)
        gid.extend([g] * len(F))
        sizes.append(len(F))
    return np.vstack(X), np.concatenate(y), np.asarray(gid, np.int32), sizes


def train_tree(pools_cache, method: str, n_estimators: int = 100, n_jobs: int = 1):
    X, y, gid, sizes = _flatten_for_trees(pools_cache)
    if method == "xgboost":
        from xgboost import XGBRanker

        m = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=42,
        )
        m.fit(X, y, qid=gid)
    elif method == "lightgbm":
        from lightgbm import LGBMRanker

        m = LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
        )
        m.fit(X, y, group=sizes)
    else:
        raise ValueError(method)
    return m


# ==========================================================================
# 6. Scoring a model over eval pools
# ==========================================================================
def _rank_with(model_kind, model, feats: torch.Tensor) -> np.ndarray:
    F = feats.numpy()
    if model_kind == "cosine":
        return np.argsort(-F[:, 0])  # col 0 = z-scored s_query
    if model_kind == "rl":
        with torch.no_grad():
            return cbr.greedy_ranking(model(feats))
    return np.argsort(-model.predict(F))  # xgboost / lightgbm


def score_model(model_kind, model, eval_cache, ks) -> dict:
    agg = {k: [] for k in ks}
    mrr_l, map_l = [], []
    for feats, gold_set, p in eval_cache:
        rank = _rank_with(model_kind, model, feats)
        for k in ks:
            agg[k].append(recall_at_k(rank, gold_set, k, p.n_gold_total))
        mrr_l.append(mrr(rank, gold_set))
        map_l.append(average_precision(rank, gold_set, p.n_gold_total))
    out = {f"R@{k}": float(np.mean(agg[k])) for k in ks}
    out["MRR"] = float(np.mean(mrr_l))
    out["MAP"] = float(np.mean(map_l))
    return out


# ==========================================================================
# 7. The sweep
# ==========================================================================
def run_topn_sweep(
    a: dict,
    ns=(50, 75, 100, 200),
    ks=(1, 5, 10, 20),
    models=("cosine", "rl", "xgboost", "lightgbm"),
    rl_epochs: int = 10,
    rl_top_m: int = 30,
    n_estimators: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """For each N: retrieve top-N globally, (re)train each learned model on the
    train top-N pools, evaluate every model on the validation top-N pools.
    Returns a tidy DataFrame: one row per (N, Model) with R@k / MRR / MAP and the
    first-stage retrieval ceiling for that N."""
    ks = list(ks)
    rows = []
    for n in ns:
        if verbose:
            print(f"\n=== N = {n} ===")
        train_pools = build_topn_pools(a, "train", n, inject_gold=True)
        val_pools = build_topn_pools(a, "validation", n, inject_gold=False)
        train_cache = _cache_features(train_pools)
        val_cache = _cache_features(val_pools)
        n_feat = train_cache[0][0].shape[1]

        ceiling = first_stage_recall(val_pools)
        if verbose:
            print(f"  first-stage recall ceiling (val): {ceiling:.4f}")

        trained = {}
        if "rl" in models:
            if verbose:
                print("  training RL...")
            trained["rl"] = train_rl(
                train_cache, n_feat, epochs=rl_epochs, top_m=rl_top_m
            )
        if "xgboost" in models:
            if verbose:
                print("  training XGBoost...")
            trained["xgboost"] = train_tree(train_cache, "xgboost", n_estimators)
        if "lightgbm" in models:
            if verbose:
                print("  training LightGBM...")
            trained["lightgbm"] = train_tree(train_cache, "lightgbm", n_estimators)

        label = {
            "cosine": "Cosine Baseline",
            "rl": "RL Contextual Bandit",
            "xgboost": "XGBRanker",
            "lightgbm": "LGBMRanker",
        }
        for kind in models:
            model = trained.get(kind)  # None for cosine
            metrics = score_model(kind, model, val_cache, ks)
            rows.append(dict(N=n, Model=label[kind], ceiling=ceiling, **metrics))

    return pd.DataFrame(rows)


def pivot_metric(sweep_df: pd.DataFrame, metric: str = "R@10") -> pd.DataFrame:
    """Models x N table for one metric, with the ceiling as a row for reference."""
    t = sweep_df.pivot(index="Model", columns="N", values=metric)
    ceil = sweep_df.drop_duplicates("N").set_index("N")["ceiling"]
    t.loc["-- first-stage ceiling --"] = ceil
    return t
