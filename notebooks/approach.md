"""
# Hierarchical Graph-RL Retrieval-Augmented Generation

## System Description and MDP Formulation

### 1. Motivation

Standard Vector RAG retrieves the globally top-$k$ documents by cosine similarity between the query and every document in the corpus. This strategy suffers from **aspect dominance**: all retrieved documents cluster around the single most salient query aspect, leaving other aspects uncovered. For multi-hop questions that require evidence from multiple topically distinct sources, this failure mode directly reduces downstream answer quality.

We propose a Reinforcement Learning agent that navigates a hierarchical document graph, explicitly selecting diverse document clusters to maximise *coverage* across query aspects within a fixed retrieval budget.

### 2. Offline Pipeline

Before any queries are processed, the corpus is structured offline.

* **Embedding:** Every document $d_i$ is encoded by a pre-trained dense encoder (BGE-base-en-v1.5) into a fixed-dimensional vector $\mathbf{e}_i \in \mathbb{R}^{768}$, then L2-normalised so that dot products equal cosine similarities.
* **Hierarchical Clustering:** Documents are grouped into a two-level hard hierarchy using a top-down K-Means approach on the normalised embeddings. First, $K_c$ Coarse Clusters are fitted on the full corpus. Each Coarse Cluster is then independently sub-clustered into Fine Clusters, yielding a total of $K_f$ Fine Clusters across the corpus. Cluster quality is validated by intra- vs. inter-cluster cosine similarity: intra-cluster similarity exceeds inter-cluster similarity at both levels, confirming semantic coherence.
* **Centroids:** Each Fine Cluster $c$ is represented by its centroid $\boldsymbol{\mu}_c$, computed as the L2-normalised mean of its member document embeddings. Centroids are the navigation targets for the RL agent.
* **Reranker:** A cross-encoder (BGE-Reranker-v2-m3) is loaded offline and applied at evaluation time to rerank the agent's retrieved candidate pool.

### 3. Retrieval Pipeline

For a query $q$ at inference time, the full pipeline is:

1. **Embed query.** Encode $q \to \mathbf{q} \in \mathbb{R}^{768}$, L2-normalised.
2. **RL navigation.** The agent selects $N_{\text{commits}}$ Fine Clusters sequentially. For each selected cluster the environment automatically extracts the top-$k$ documents by cosine similarity to $\mathbf{q}$, building a candidate pool of $N_{\text{commits}} \times k$ documents.
3. **Rerank.** The cross-encoder scores every (query, document) pair in the candidate pool and returns the top-20 documents as the final retrieval result.

The baseline (Vector RAG) follows the same pipeline but replaces Step 2 with global top-$(N_{\text{commits}} \times k)$ cosine retrieval, so both systems enter the reranker with the same pool size under identical conditions.

### 4. Markov Decision Process Formulation

#### Episode Structure

Each query induces one fixed-horizon episode of exactly $T = N_{\text{commits}}$ steps. At each step the agent commits to one Fine Cluster; the environment immediately extracts $k$ documents from it. The episode terminates when all $T$ commits are exhausted.

#### State Space

The state at step $t$ is a fixed-size vector $\mathbf{s}_t$ that gives the agent complete information about the query, its current position, and its retrieval history:

$$\mathbf{s}_t = \bigl[ \underbrace{\mathbf{r}_t}_{\text{radar}}\ \big|\ \underbrace{\mathbf{m}_t}_{\text{fine mask}}\ \big|\ \underbrace{b_t}_{\text{budget}}\bigr] \in \mathbb{R}^{2K_f + 1}$$

| Component | Dim | Description |
| --- | --- | --- |
| **Radar $\mathbf{r}_t$** | $K_f$ | Cosine similarity between query $\mathbf{q}$ and every Fine Cluster centroid: $r_{t,c} = \mathbf{q}^\top \boldsymbol{\mu}_c$. Gives the agent a global view of which clusters are most relevant. |
| **Fine mask $\mathbf{m}_t$** | $K_f$ | Binary vector. $m_{t,c} = 1$ if cluster $c$ has already been committed in this episode, $0$ otherwise. Encodes full retrieval history. |
| **Budget $b_t$** | $1$ | Fraction of budget remaining: $b_t = 1 - t/T$. Allows the agent to adjust its strategy near the end of the episode. |

#### Action Space

At each step $t$ the agent selects one Fine Cluster index: $a_t \in \{0, 1, \ldots, K_f - 1\}$

* **Action masking:** To prevent revisiting clusters and to focus exploration, two masks are applied before the softmax:
* **Committed mask:** Already-committed clusters are set to $-\infty$ in the logits.
* **Top-$K$ pre-filter:** Only the top-20 Fine Clusters by radar similarity $r_{t,c}$ are available. Clusters ranked below 20 are masked to $-\infty$.



#### Transition Dynamics

When the agent selects cluster $a_t = c$:

1. Compute cosine similarity between $\mathbf{q}$ and all documents $d_i \in c$: $\text{sim}(i) = \mathbf{q}^\top \mathbf{e}_i$.
2. Extract the top-$k$ documents not already in the retrieval pool.
3. Add them to the pool; update $\mathbf{m}_{t+1}$ and $b_{t+1}$.

#### Reward Signal

The agent receives **dense step rewards**. At each step $t$, the reward is the marginal increase in recall caused by committing to cluster $a_t$:

$$r_t = \frac{\bigl|\mathcal{R}_t \cap \mathcal{G}\bigr| - \bigl|\mathcal{R}_{t-1} \cap \mathcal{G}\bigr|}{|\mathcal{G}|}$$

Discounted returns are computed with $\gamma = 0.99$ over the episode: $G_t = \sum_{t'=t}^{T} \gamma^{t'-t}\, r_{t'}$

### 5. Policy and Training

#### Actor-Critic Architecture

A two-headed MLP shares a common trunk with Layer Normalisation and ReLU activations. Weights are initialised orthogonally.

* **Actor head:** $\pi_\theta(a \mid \mathbf{s}_t) = \text{softmax}(\mathbf{W}_a \mathbf{h} + \text{mask})$
* **Critic head:** $V_\phi(\mathbf{s}_t) = \mathbf{w}_v^\top \mathbf{h}$

#### REINFORCE with Baseline

The policy is trained with the Actor-Critic algorithm, normalising advantages per batch. The entropy bonus $\mathcal{H}$ prevents premature convergence to Vector RAG behaviour.

| Hyperparameter | Value |
| --- | --- |
| Commits per episode ($T$) | 10 |
| Docs per commit ($k$) | 5 |
| Learning rate | $3 \times 10^{-4}$ |
| Entropy coefficient ($\lambda_e$) | 0.02 |
| Discount ($\gamma$) | 0.99 |

### 6. Evaluation Protocol

Both systems are evaluated by:

1. **Retrieve** 50 candidates.
2. **Rerank** with BGE-Reranker-v2-m3.
3. **Truncate** to top-20.
4. **Score** Recall@20 against ground-truth.

"""