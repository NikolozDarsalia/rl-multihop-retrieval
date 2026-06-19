"""Option A - Contextual-bandit reranker for the per-question candidate pool.

We treat reranking the (<=20) candidates of a single question as a contextual
bandit: the context is (query vector + candidate vectors), the action is a
ranking of the candidates, and the reward is a retrieval metric (default: mean
average precision over the gold chain). This is the BanditRank formulation
(Pang et al., 2019) - a contextual bandit is just an MDP with episode length 1,
trained with policy gradient to directly optimize the discrete IR metric we
actually care about, instead of a surrogate loss.

Algorithm: self-critical REINFORCE (a.k.a. self-critical sequence training).
For each query we sample a ranking from a Plackett-Luce distribution over the
policy's scores, and use the *greedy* ranking's reward as the baseline. This
baseline is per-query, so it sidesteps the high-variance-across-difficulty
problem that Das et al. (2019) reported for batch-mean baselines in QA.

IMPORTANT - this is the deliberately "embedding only" reranker. Its features
are all derived from the BGE vectors you already have:

    s_query    cosine(candidate, query)            -- the raw Vector-RAG signal
    s_prf      cosine(candidate, top-1-by-cosine)  -- pseudo-relevance feedback;
                                                       this is the single-step
                                                       approximation of the
                                                       bridge signal your EDA
                                                       found (later-hop golds sit
                                                       near the first-hop gold)
    s_centroid cosine(candidate, pool centroid)    -- how "typical" the candidate
                                                       is for this pool
    s_maxother max cosine to other candidates      -- redundancy / near-duplicate
    rank_q     normalized cosine rank              -- positional prior
    log_freq   (optional) log corpus frequency     -- only if nodes 'freq' given

With only s_query the policy can do no better than cosine (it is a monotone
transform). The other features - especially s_prf - are what let it beat the
baseline, and they are exactly the levers your multi-hop diagnosis points at.

Nothing here uses an LLM, a knowledge graph, an agent, or the clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURE_NAMES = ["s_query", "s_prf", "s_centroid", "s_maxother", "rank_q"]
DEFAULT_KS = list(range(1, 21))


# ==========================================================================
# 1. Artifact loading (mirrors the `a = load_artifacts()` convention you use
#    in notebook 03; safe to import without your config/paths being present)
# ==========================================================================
def load_artifacts(artifacts_dir=None, interim_dir=None) -> dict:
    """Load embeddings + tables into the `a` dict used across the notebooks.

    Returns keys: doc_emb, query_emb, query_ids, questions, nodes,
    node_id_to_idx, query_id_to_idx. Paths default to the values in
    src/config.py; pass explicit dirs to override.
    """
    import json

    if artifacts_dir is None or interim_dir is None:
        from . import config  # lazy: only needed for the default paths

        artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
        interim_dir = interim_dir or config.INTERIM_DIR

    from pathlib import Path

    artifacts_dir, interim_dir = Path(artifacts_dir), Path(interim_dir)

    doc_emb = np.load(artifacts_dir / "doc_emb.npy")
    query_emb = np.load(artifacts_dir / "query_emb.npy")
    query_ids = np.load(artifacts_dir / "query_ids.npy", allow_pickle=False).astype(str)
    nodes = pd.read_parquet(interim_dir / "nodes.parquet")
    questions = pd.read_parquet(interim_dir / "questions.parquet")

    manifest_path = artifacts_dir / "embeddings_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    return build_artifact_dict(
        doc_emb, query_emb, query_ids, questions, nodes, manifest
    )


def build_artifact_dict(
    doc_emb, query_emb, query_ids, questions, nodes=None, manifest=None
) -> dict:
    """Assemble the `a` dict from in-memory arrays (used by the test harness)."""
    query_ids = np.asarray(query_ids).astype(str)
    a = dict(
        doc_emb=np.ascontiguousarray(doc_emb, dtype=np.float32),
        query_emb=np.ascontiguousarray(query_emb, dtype=np.float32),
        query_ids=query_ids,
        questions=questions,
        nodes=nodes,
        manifest=manifest or {},
        query_id_to_idx={qid: i for i, qid in enumerate(query_ids)},
    )
    # node_id == row index by construction (see embeddings.py), but we keep the
    # explicit map so the contract is visible and overridable.
    if nodes is not None:
        a["node_id_to_idx"] = {int(n): int(n) for n in nodes["node_id"].to_numpy()}
    else:
        a["node_id_to_idx"] = {i: i for i in range(len(doc_emb))}
    return a


# ==========================================================================
# 2. Per-query pools: turn one question into (query_vec, cand_vecs, gold_mask)
# ==========================================================================
@dataclass
class QueryPool:
    question_id: str
    n_hops: int
    query_vec: np.ndarray  # [d]
    cand_vecs: np.ndarray  # [n_cand, d]
    cand_node_ids: np.ndarray  # [n_cand]
    gold_local: np.ndarray  # indices into cand_* that are gold (hop order)


def iter_query_pools(a: dict, split: str | None = None):
    """Yield a QueryPool per question (optionally filtered to a split)."""
    q = a["questions"]
    if split is not None:
        q = q[q["split"] == split]
    doc_emb, query_emb = a["doc_emb"], a["query_emb"]
    q2i, n2i = a["query_id_to_idx"], a["node_id_to_idx"]

    for _, row in q.iterrows():
        qid = str(row["question_id"])
        if qid not in q2i:
            continue
        cand_ids = np.asarray(row["candidate_node_ids"])
        gold_ids = list(row["gold_node_ids"])
        cand_rows = np.array([n2i[int(c)] for c in cand_ids])
        cand_vecs = doc_emb[cand_rows]
        # local positions of gold within the candidate list, kept in hop order
        id_to_local = {int(c): j for j, c in enumerate(cand_ids)}
        gold_local = np.array(
            [id_to_local[int(g)] for g in gold_ids if int(g) in id_to_local]
        )
        yield QueryPool(
            question_id=qid,
            n_hops=int(row["n_hops"]),
            query_vec=query_emb[q2i[qid]],
            cand_vecs=cand_vecs,
            cand_node_ids=cand_ids,
            gold_local=gold_local,
        )


# ==========================================================================
# 3. Feature extraction (embedding-only)
# ==========================================================================
def _l2(x, axis=-1, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def build_features(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    freq: np.ndarray | None = None,
    drop_features: set | None = None,
) -> np.ndarray:
    """Return per-candidate feature matrix [n_cand, n_feat], z-scored within the pool.

    All features are functions of the embeddings (and optionally corpus freq).

    drop_features: names to ablate (e.g. {"s_prf"}). The column is zeroed BEFORE
    z-scoring, so it becomes a constant 0 that carries no information, while the
    matrix width and the col-0 == s_query invariant stay unchanged.
    """
    q = _l2(query_vec.astype(np.float32))
    C = _l2(cand_vecs.astype(np.float32), axis=1)
    n = len(C)

    s_query = C @ q  # cosine to query
    top1 = int(np.argmax(s_query))
    s_prf = C @ C[top1]  # pseudo-relevance feedback
    centroid = _l2(C.mean(axis=0))
    s_centroid = C @ centroid
    sim_matrix = C @ C.T
    np.fill_diagonal(sim_matrix, -np.inf)
    s_maxother = sim_matrix.max(axis=1) if n > 1 else np.zeros(n, dtype=np.float32)
    # normalized cosine rank in [0,1] (1 == best by cosine)
    order = np.argsort(-s_query)
    rank_q = np.empty(n, dtype=np.float32)
    rank_q[order] = np.linspace(1.0, 0.0, n) if n > 1 else np.array([1.0])

    named = {
        "s_query": s_query,
        "s_prf": s_prf,
        "s_centroid": s_centroid,
        "s_maxother": s_maxother,
        "rank_q": rank_q,
    }
    if freq is not None:
        named["log_freq"] = np.log1p(freq.astype(np.float32))
    if drop_features:
        for name in drop_features:
            if name in named:
                named[name] = np.zeros(n, dtype=np.float32)  # ablated -> constant 0

    ordered = FEATURE_NAMES + (["log_freq"] if freq is not None else [])
    feats = [named[name] for name in ordered]

    F = np.stack(feats, axis=1).astype(np.float32)
    # z-score each column within this pool so the MLP trains on comparable scales
    mu, sd = F.mean(axis=0, keepdims=True), F.std(axis=0, keepdims=True)
    return (F - mu) / np.maximum(sd, 1e-6)


def feature_names(include_freq: bool) -> list[str]:
    return FEATURE_NAMES + (["log_freq"] if include_freq else [])


# ==========================================================================
# 4. Metrics (computed from a ranking = ordered list of candidate indices)
# ==========================================================================
def recall_at_k(ranking: np.ndarray, gold_local: np.ndarray, k: int) -> float:
    if len(gold_local) == 0:
        return 0.0
    topk = set(ranking[:k].tolist())
    return len(topk & set(gold_local.tolist())) / len(gold_local)


def mrr(ranking: np.ndarray, gold_local: np.ndarray) -> float:
    gold = set(gold_local.tolist())
    for r, idx in enumerate(ranking, start=1):
        if int(idx) in gold:
            return 1.0 / r
    return 0.0


def average_precision(ranking: np.ndarray, gold_local: np.ndarray) -> float:
    """MAP-style reward: dense signal at every gold position (BanditRank used MAP)."""
    gold = set(gold_local.tolist())
    if not gold:
        return 0.0
    hits, score = 0, 0.0
    for r, idx in enumerate(ranking, start=1):
        if int(idx) in gold:
            hits += 1
            score += hits / r
    return score / len(gold)


REWARD_FNS = {"map": average_precision, "mrr": mrr}


# ==========================================================================
# 5. Policy + Plackett-Luce sampling
# ==========================================================================
class RerankPolicy(nn.Module):
    """Small MLP that scores each candidate independently (pointwise scorer)."""

    def __init__(self, n_feat: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, feats: torch.Tensor
    ) -> torch.Tensor:  # [n_cand, n_feat] -> [n_cand]
        return self.net(feats).squeeze(-1)


def sample_plackett_luce(scores: torch.Tensor, temperature: float = 1.0):
    """Sample a full ranking by sampling without replacement from softmax(scores).

    Returns (ranking LongTensor [n], log_prob scalar tensor).
    log_prob is the sum of log-softmax of each chosen item among the remaining.
    """
    s = scores / temperature
    remaining = list(range(len(s)))
    ranking, logp = [], scores.new_zeros(())
    for _ in range(len(s)):
        sub = s[remaining]
        probs = torch.softmax(sub, dim=0)
        choice = torch.multinomial(probs, 1).item()
        logp = logp + torch.log(probs[choice] + 1e-12)
        ranking.append(remaining.pop(choice))
    return torch.tensor(ranking, dtype=torch.long), logp


def greedy_ranking(scores: torch.Tensor) -> np.ndarray:
    return torch.argsort(scores, descending=True).cpu().numpy()


# ==========================================================================
# 6. Training (self-critical REINFORCE)
# ==========================================================================
@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 1e-3
    hidden: int = 32
    temperature: float = 1.0
    reward: str = "map"  # "map" or "mrr"
    batch_size: int = 32  # queries accumulated per optimizer step
    use_freq: bool = False
    seed: int = 199806
    log_every: int = 1
    entropy_coef: float = 0.0  # optional exploration bonus


def _precompute(pools, a, use_freq, drop_features=None):
    """Cache (features, gold_local) per pool so we don't recompute every epoch."""
    freq_map = None
    if use_freq and a.get("nodes") is not None:
        freq_map = dict(
            zip(a["nodes"]["node_id"].astype(int), a["nodes"]["freq"].astype(float))
        )
    cache = []
    for p in pools:
        freq = None
        if freq_map is not None:
            freq = np.array([freq_map.get(int(c), 0.0) for c in p.cand_node_ids])
        F = build_features(
            p.query_vec, p.cand_vecs, freq=freq, drop_features=drop_features
        )
        cache.append((torch.from_numpy(F), p.gold_local, p))
    return cache


def train_reranker(
    a: dict, cfg: TrainConfig = None, verbose: bool = True, drop_features=None
):
    """Train the bandit reranker on the train split. Returns (policy, history)."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_pools = list(iter_query_pools(a, split="train"))
    cache = _precompute(train_pools, a, cfg.use_freq, drop_features=drop_features)
    n_feat = cache[0][0].shape[1]
    reward_fn = REWARD_FNS[cfg.reward]

    policy = RerankPolicy(n_feat, hidden=cfg.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    history = []
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(cache))
        epoch_adv, epoch_r = [], []
        opt.zero_grad()
        batch_loss, in_batch = policy_zero(), 0

        for step, i in enumerate(order, start=1):
            feats, gold_local, _ = cache[i]
            scores = policy(feats)
            ranking, logp = sample_plackett_luce(scores, cfg.temperature)
            r_sample = reward_fn(ranking.numpy(), gold_local)
            r_greedy = reward_fn(greedy_ranking(scores.detach()), gold_local)
            advantage = r_sample - r_greedy  # self-critical baseline

            loss = -advantage * logp
            if cfg.entropy_coef > 0:
                probs = torch.softmax(scores, dim=0)
                loss = loss - cfg.entropy_coef * (
                    -(probs * torch.log(probs + 1e-12)).sum()
                )
            batch_loss = batch_loss + loss
            in_batch += 1
            epoch_adv.append(advantage)
            epoch_r.append(r_greedy)

            if in_batch == cfg.batch_size or step == len(order):
                (batch_loss / in_batch).backward()
                opt.step()
                opt.zero_grad()
                batch_loss, in_batch = policy_zero(), 0

        hist = dict(
            epoch=epoch,
            mean_greedy_reward=float(np.mean(epoch_r)),
            mean_advantage=float(np.mean(epoch_adv)),
        )
        history.append(hist)
        if verbose and (epoch % cfg.log_every == 0):
            print(
                f"epoch {epoch:>2}  greedy_{cfg.reward}={hist['mean_greedy_reward']:.4f}"
                f"  mean_adv={hist['mean_advantage']:+.4f}"
            )

    return policy, pd.DataFrame(history)


def policy_zero():
    return torch.zeros((), requires_grad=False)


# ==========================================================================
# 7. Evaluation: policy vs cosine baseline, cumulative Recall@K
# ==========================================================================
@torch.no_grad()
def evaluate(
    a: dict,
    policy: RerankPolicy | None,
    split: str = "validation",
    ks=DEFAULT_KS,
    use_freq: bool = False,
    by_hop: bool = False,
    drop_features=None,
) -> pd.DataFrame:
    """Cumulative Recall@K (and MRR/MAP) for the cosine baseline and, if a policy
    is given, the learned reranker. Set by_hop=True to break recall out per n_hops.
    """
    pools = list(iter_query_pools(a, split=split))
    cache = _precompute(pools, a, use_freq, drop_features=drop_features)

    rows = []
    groups = {"all": cache}
    if by_hop:
        groups = {}
        for item in cache:
            groups.setdefault(item[2].n_hops, []).append(item)

    for gname, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        agg = {("cosine", k): [] for k in ks}
        agg.update({("reranker", k): [] for k in ks})
        mrr_c, mrr_r, map_c, map_r = [], [], [], []

        for feats, gold_local, pool in items:
            # cosine baseline ranking = sort by s_query (col 0 of features is z-scored s_query;
            # z-scoring is monotone so argsort is identical to raw cosine order)
            cos_rank = torch.argsort(feats[:, 0], descending=True).numpy()
            for k in ks:
                agg[("cosine", k)].append(recall_at_k(cos_rank, gold_local, k))
            mrr_c.append(mrr(cos_rank, gold_local))
            map_c.append(average_precision(cos_rank, gold_local))

            if policy is not None:
                scores = policy(feats)
                r_rank = greedy_ranking(scores)
                for k in ks:
                    agg[("reranker", k)].append(recall_at_k(r_rank, gold_local, k))
                mrr_r.append(mrr(r_rank, gold_local))
                map_r.append(average_precision(r_rank, gold_local))

        for k in ks:
            row = dict(
                group=gname, K=k, recall_cosine=float(np.mean(agg[("cosine", k)]))
            )
            if policy is not None:
                row["recall_reranker"] = float(np.mean(agg[("reranker", k)]))
                row["delta"] = row["recall_reranker"] - row["recall_cosine"]
            rows.append(row)

        # attach summary scalars on the K==max row for convenience
        summ = dict(group=gname, K="MRR", recall_cosine=float(np.mean(mrr_c)))
        if policy is not None:
            summ["recall_reranker"] = float(np.mean(mrr_r))
            summ["delta"] = summ["recall_reranker"] - summ["recall_cosine"]
        rows.append(summ)
        summ = dict(group=gname, K="MAP", recall_cosine=float(np.mean(map_c)))
        if policy is not None:
            summ["recall_reranker"] = float(np.mean(map_r))
            summ["delta"] = summ["recall_reranker"] - summ["recall_cosine"]
        rows.append(summ)

    return pd.DataFrame(rows)


# ==========================================================================
# 8. Persistence
# ==========================================================================
def save_policy(
    policy: RerankPolicy, path, cfg: TrainConfig = None, n_feat: int = None
):
    import json, os

    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    torch.save(policy.state_dict(), str(path))
    if cfg is not None:
        meta = dict(
            hidden=cfg.hidden,
            reward=cfg.reward,
            n_feat=n_feat,
            features=feature_names(cfg.use_freq),
        )
        with open(str(path) + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def load_policy(path, n_feat: int, hidden: int = 32) -> RerankPolicy:
    policy = RerankPolicy(n_feat, hidden=hidden)
    policy.load_state_dict(torch.load(str(path)))
    policy.eval()
    return policy


# ==========================================================================
# 9. Tree-based rankers (LambdaMART via XGBoost / LightGBM)  [FIXED]
# ==========================================================================
# NOTE on the macOS kernel crash: torch, xgboost, lightgbm and sentence-
# transformers each ship their own OpenMP runtime. Loading more than one in the
# same kernel aborts the process on Apple Silicon. Put this at the TOP of the
# notebook's first cell, BEFORE importing this module (which imports torch):
#
#     import os
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # tolerate >1 OpenMP runtime
#     os.environ["OMP_NUM_THREADS"] = "1"           # avoid thread oversubscription
#
# then restart the kernel and run from the top.
def _prepare_tree_data(a: dict, split: str, use_freq: bool = False, drop_features=None):
    """Flatten per-query pools into tabular LTR format.

    Returns X [N,F], y [N] (1=gold), group_ids [N] (INTEGER, non-decreasing,
    one id per pool - required by XGBoost's qid path), and group_sizes (list of
    per-pool candidate counts - required by LightGBM's group path).
    """
    pools = list(iter_query_pools(a, split=split))
    cache = _precompute(pools, a, use_freq, drop_features=drop_features)

    X, y, group_ids, group_sizes = [], [], [], []
    for gid, (feats, gold_local, pool) in enumerate(cache):
        feats_np = feats.numpy()
        n_cands = len(feats_np)
        X.append(feats_np)
        labels = np.zeros(n_cands, dtype=int)
        labels[gold_local] = 1
        y.append(labels)
        group_ids.extend([gid] * n_cands)  # integer, increments per pool
        group_sizes.append(n_cands)

    return (
        np.vstack(X),
        np.concatenate(y),
        np.asarray(group_ids, dtype=np.int32),
        group_sizes,
    )


def train_tree_ranker(
    a: dict,
    method: str = "xgboost",
    use_freq: bool = False,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    n_jobs: int = 1,
    drop_features=None,
):
    """Train a supervised LambdaMART ranker on the geometric features.

    Uses integer group ids (XGBoost) / group sizes (LightGBM). n_jobs=1 keeps
    OpenMP thread usage minimal, which also reduces crash risk on macOS.
    """
    X, y, group_ids, group_sizes = _prepare_tree_data(
        a, "train", use_freq, drop_features=drop_features
    )

    if method.lower() == "xgboost":
        from xgboost import XGBRanker

        model = XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=42,
        )
        model.fit(X, y, qid=group_ids)  # integer qid, non-decreasing

    elif method.lower() == "lightgbm":
        from lightgbm import LGBMRanker

        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y, group=group_sizes)  # per-pool sizes

    else:
        raise ValueError("method must be 'xgboost' or 'lightgbm'")

    return model


def evaluate_tree_ranker(
    a: dict,
    model,
    split: str = "validation",
    ks=DEFAULT_KS,
    use_freq: bool = False,
    model_name: str | None = None,
    drop_features=None,
) -> pd.DataFrame:
    """Evaluate a trained tree model over the candidate pools."""
    pools = list(iter_query_pools(a, split=split))
    cache = _precompute(pools, a, use_freq, drop_features=drop_features)
    name = model_name or type(model).__name__

    agg = {k: [] for k in ks}
    mrr_r, map_r = [], []
    for feats, gold_local, pool in cache:
        scores = model.predict(feats.numpy())
        r_rank = np.argsort(-scores)
        for k in ks:
            agg[k].append(recall_at_k(r_rank, gold_local, k))
        mrr_r.append(mrr(r_rank, gold_local))
        map_r.append(average_precision(r_rank, gold_local))

    rows = [{"Model": name, "K": k, "Score": float(np.mean(agg[k]))} for k in ks]
    rows.append({"Model": name, "K": "MRR", "Score": float(np.mean(mrr_r))})
    rows.append({"Model": name, "K": "MAP", "Score": float(np.mean(map_r))})
    return pd.DataFrame(rows)


# ==========================================================================
# 10. Cross-encoder baseline (the heavyweight)  [FIXED]
# ==========================================================================
@torch.no_grad()
def evaluate_cross_encoder(
    a: dict,
    model_name: str = "BAAI/bge-reranker-base",
    split: str = "validation",
    ks=DEFAULT_KS,
    max_length: int = 256,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Rerank each pool with a HF cross-encoder over raw text.

    Reads the correct schema columns: questions.query_text and nodes.title/text.
    Docs are formatted as "title. text" to match how 03 built the embeddings.
    """
    from sentence_transformers import CrossEncoder

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    ce = CrossEncoder(model_name, device=device, max_length=max_length)

    q_df = a["questions"].set_index("question_id")
    n_df = a["nodes"].set_index("node_id")

    pools = list(iter_query_pools(a, split=split))
    agg = {k: [] for k in ks}
    mrr_r, map_r = [], []

    for pool in pools:
        query_text = str(q_df.loc[pool.question_id, "query_text"])  # FIXED column
        cand_texts = []
        for nid in pool.cand_node_ids:
            row = n_df.loc[int(nid)]
            title = str(row.get("title", "") or "")
            text = str(row.get("text", "") or "")
            cand_texts.append((title + ". " + text).strip())
        scores = ce.predict(
            [(query_text, d) for d in cand_texts],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        r_rank = np.argsort(-np.asarray(scores))
        for k in ks:
            agg[k].append(recall_at_k(r_rank, pool.gold_local, k))
        mrr_r.append(mrr(r_rank, pool.gold_local))
        map_r.append(average_precision(r_rank, pool.gold_local))

    rows = [
        {"Model": "CrossEncoder", "K": k, "Score": float(np.mean(agg[k]))} for k in ks
    ]
    rows.append({"Model": "CrossEncoder", "K": "MRR", "Score": float(np.mean(mrr_r))})
    rows.append({"Model": "CrossEncoder", "K": "MAP", "Score": float(np.mean(map_r))})
    return pd.DataFrame(rows)
