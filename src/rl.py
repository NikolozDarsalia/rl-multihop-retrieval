"""
Design
------
  - Agent makes COMMITS_PER_EPISODE decisions per episode
  - Each decision = pick one fine cluster (from Top-20 pre-filtered candidates)
  - Environment auto-extracts TOP_K_PER_COMMIT docs from that cluster
    using cosine similarity to query (the reranker)
  - Total docs retrieved = COMMITS_PER_EPISODE * TOP_K_PER_COMMIT
  - Dense step rewards = marginal increase in recall at each step
  - Eval: Agent (Pool of 50) -> Reranker -> Top-20 Recall
"""

from __future__ import annotations
import argparse
import json
import time
import random
import pickle
import joblib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from FlagEmbedding import FlagReranker

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path("..")
ARTIFACTS = ROOT / "data" / "artifacts"
INTERIM   = ROOT / "data" / "interim"
CKPT_DIR  = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
COMMITS_PER_EPISODE = 10      
TOP_K_PER_COMMIT    = 5       
TOTAL_RETRIEVED     = COMMITS_PER_EPISODE * TOP_K_PER_COMMIT  # 50
FINAL_K             = 20      

EMB_DIM      = 768
HIDDEN       = 512
LR           = 3e-4
BATCH_EPS    = 32             
N_EPISODES   = 20_000
ENTROPY_COEF = 0.02
CRITIC_COEF  = 0.5
GRAD_CLIP    = 1.0
EVAL_EVERY   = 2_000 
SEED         = 199806

# ── load artifacts ────────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    print("Loading artifacts ...")
    a = {}
    CLUSTER_DIR = ARTIFACTS / "clustering_output"

    a["doc_emb"]       = normalize(np.load(ARTIFACTS / "doc_emb.npy").astype(np.float32), norm="l2")
    a["query_emb"]     = normalize(np.load(ARTIFACTS / "query_emb.npy").astype(np.float32), norm="l2")
    a["query_ids"]     = np.load(ARTIFACTS / "query_ids.npy", allow_pickle=True)
    a["questions"]     = pd.read_parquet(INTERIM / "questions.parquet")
    
    nodes_df           = pd.read_parquet(INTERIM / "nodes.parquet")
    a["doc_texts"]     = (nodes_df["title"] + " " + nodes_df["text"]).tolist()

    raw_labels = np.load(CLUSTER_DIR / "doc_hierarchical_labels.npy", allow_pickle=True)
    raw_fine_labels = raw_labels[:, 1] if raw_labels.ndim > 1 else raw_labels
    
    if isinstance(raw_fine_labels[0], str):
        unique_labels = np.unique(raw_fine_labels)
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        a["labels_fine"] = np.array([label_to_id[lbl] for lbl in raw_fine_labels])
    else:
        a["labels_fine"] = raw_fine_labels

    n_fine = len(np.unique(a["labels_fine"]))
    a["n_fine"] = n_fine
    a["cluster_members"] = [np.where(a["labels_fine"] == c)[0] for c in range(n_fine)]
    
    centroids = [a["doc_emb"][a["cluster_members"][c]].mean(axis=0) if len(a["cluster_members"][c]) > 0 else np.zeros(768) for c in range(n_fine)]
    a["fine_cent"] = normalize(np.vstack(centroids).astype(np.float32), norm="l2")
    a["qid_to_qi"] = {qid: i for i, qid in enumerate(a["query_ids"])}

    return a

# ── reranker ──────────────────────────────────────────────────────────────────

def rerank(query: str, doc_ids: list[int], a: dict, reranker, top_k: int = 20) -> list[int]:
    if len(doc_ids) <= top_k: 
        return doc_ids
        
    # The FlagReranker object usually exposes its internal tokenizer
    tokenizer = reranker.tokenizer
    
    # We prepare the pairs manually to avoid the broken internal method
    # and then pass the result to the reranker's compute_score
    pairs = [(query, a["doc_texts"][did]) for did in doc_ids]
    
    # The compute_score method typically handles the tokenization internally;
    # if it continues to error, we use the library's official inference mode
    scores = reranker.compute_score(pairs, normalize=True)
    
    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [did for did, _ in ranked[:top_k]]

def extract_top_k(cluster_id, query_norm, a, k, exclude):
    members = a["cluster_members"][cluster_id]
    if len(members) == 0: return []
    sims = a["doc_emb"][members] @ query_norm
    if exclude:
        for i, m in enumerate(members):
            if int(m) in exclude: sims[i] = -2.0
    return [int(members[i]) for i in np.argsort(sims)[::-1][:k]]

# ── environment ───────────────────────────────────────────────────────────────

class Episode:
    def __init__(self, row, qi, a):
        self.gold, self.a = set(row["gold_node_ids"]), a
        self.query_norm = a["query_emb"][qi]
        self.q_fine_sims = (a["fine_cent"] @ self.query_norm).astype(np.float32)
        self.valid_candidates = np.argsort(self.q_fine_sims)[::-1][:min(20, a["n_fine"])].tolist()
        self.current_fine = int(self.valid_candidates[0])
        self.committed, self.retrieved, self.retrieved_set, self.step, self.done = [], [], set(), 0, False

    def state(self):
        q_sims = self.q_fine_sims
        mask = np.zeros(self.a["n_fine"], dtype=np.float32)
        for c in self.committed: mask[c] = 1.0
        budget = np.array([1.0 - self.step / COMMITS_PER_EPISODE], dtype=np.float32)
        return np.concatenate([self.query_norm, self.a["fine_cent"][self.current_fine], 
                               [float(self.q_fine_sims[self.current_fine])], q_sims, mask, budget])

    def action_mask(self):
        mask = np.zeros(self.a["n_fine"], dtype=bool)
        mask[self.valid_candidates] = True
        for c in self.committed: mask[c] = False
        return mask

    def step_env(self, fine_cluster_id):
        self.committed.append(fine_cluster_id)
        self.current_fine = fine_cluster_id
        prev_hits = sum(1 for d in self.retrieved if d in self.gold)
        docs = extract_top_k(fine_cluster_id, self.query_norm, self.a, TOP_K_PER_COMMIT, self.retrieved_set)
        self.retrieved.extend(docs); self.retrieved_set.update(docs); self.step += 1
        if self.step >= COMMITS_PER_EPISODE: self.done = True
        return (sum(1 for d in self.retrieved if d in self.gold) - prev_hits) / len(self.gold) if self.gold else 0.0

    def reward(self):
        return sum(1 for d in self.retrieved if d in self.gold) / len(self.gold) if self.gold else 0.0

# ── model + training ──────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, HIDDEN // 2), nn.LayerNorm(HIDDEN // 2), nn.ReLU())
        self.actor, self.critic = nn.Linear(HIDDEN // 2, n_actions), nn.Linear(HIDDEN // 2, 1)

    def forward(self, x, mask=None):
        h = self.shared(x)
        logits = self.actor(h).masked_fill(~mask, float("-inf")) if mask is not None else self.actor(h)
        return F.softmax(logits, dim=-1), self.critic(h).squeeze(-1)

def run_episode(model, row, qi, a, device, greedy=False):
    ep = Episode(row, qi, a)
    states, log_probs, values, step_rewards = [], [], [], []
    while not ep.done:
        s = torch.tensor(ep.state(), dtype=torch.float32, device=device).unsqueeze(0)
        amask = torch.tensor(ep.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
        probs, val = model(s, amask)
        dist = torch.distributions.Categorical(probs)
        act = probs.argmax(dim=-1) if greedy else dist.sample()
        states.append(s.squeeze(0)); log_probs.append(dist.log_prob(act)); values.append(val.squeeze(0))
        step_rewards.append(ep.step_env(int(act.item())))
    
    G, returns = 0.0, []
    for r in reversed(step_rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    return states, log_probs, values, returns, ep.reward()

def collect_batch(model, a, rows, qis, device):
    all_lp, all_vals, all_rets, ep_rets = [], [], [], []
    for row, qi in zip(rows, qis):
        _, lps, vals, step_rets, R = run_episode(model, row, qi, a, device)
        ep_rets.append(R)
        for lp, v, r in zip(lps, vals, step_rets):
            all_lp.append(lp); all_vals.append(v); all_rets.append(torch.tensor(r, dtype=torch.float32, device=device))
    return {"log_probs": torch.stack(all_lp), "values": torch.stack(all_vals), "returns": torch.stack(all_rets), "ep_returns": torch.tensor(ep_rets)}

def update(model, opt, batch):
    lp, vals, returns = batch["log_probs"], batch["values"], batch["returns"]
    adv = (returns - vals.detach()); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = -(lp * adv).mean() + CRITIC_COEF * F.mse_loss(vals, returns) - ENTROPY_COEF * (-lp.mean())
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
    return {"loss": loss.item()}

def evaluate(model, a, reranker, n=500, split="validation", device="cpu"):
    qs = a["questions"]; val = qs[qs["split"] == split].sample(n=min(n, (qs["split"] == split).sum()), random_state=SEED)
    recalls = []
    model.eval()
    with torch.no_grad():
        for _, row in val.iterrows():
            if row["question_id"] not in a["qid_to_qi"]: continue
            _, _, _, _, _ = run_episode(model, row, a["qid_to_qi"][row["question_id"]], a, device, greedy=True)
            # Re-retrieve pool
            ep = Episode(row, a["qid_to_qi"][row["question_id"]], a)
            while not ep.done:
                s = torch.tensor(ep.state(), dtype=torch.float32, device=device).unsqueeze(0)
                amask = torch.tensor(ep.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                probs, _ = model(s, amask)
                ep.step_env(int(probs.argmax(dim=-1).item()))
            final = rerank(row["query_text"], ep.retrieved, a, reranker, top_k=FINAL_K)
            gold = set(row["gold_node_ids"])
            recalls.append(len(set(final) & gold) / len(gold) if gold else 0.0)
    return {"recall": float(np.mean(recalls)) * 100}

def train(device="cpu"):
    a = load_artifacts()
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    state_dim = EMB_DIM + EMB_DIM + 1 + a["n_fine"] + a["n_fine"] + 1
    model = ActorCritic(state_dim, a["n_fine"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    train_qs = a["questions"][a["questions"]["split"] == "train"].reset_index(drop=True)
    for ep_i in range(0, N_EPISODES, BATCH_EPS):
        batch = collect_batch(model, a, [r for _, r in train_qs.sample(n=BATCH_EPS, random_state=ep_i).iterrows()], 
                              [a["qid_to_qi"].get(r["question_id"], 0) for _, r in train_qs.sample(n=BATCH_EPS, random_state=ep_i).iterrows()], device)
        stats = update(model, opt, batch)
        if ep_i % EVAL_EVERY == 0:
            evl = evaluate(model, a, reranker, n=500, device=device)
            print(f"Ep {ep_i} | Recall: {evl['recall']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(device=args.device)
