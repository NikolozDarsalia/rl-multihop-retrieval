## 📂 1. The Interim Folder (`data/interim/`)
The **Interim** folder holds the clean, structured text dataframes. These are the human-readable building blocks of the project.

* ### `nodes.parquet`
    * **What it is:** The master text corpus containing every individual knowledge segment (document or paragraph) available in your system.
    * **What's inside:** It contains columns for text segments (`paragraph_text`), identifiers (`node_id`), and their parent document or Wikipedia subject labels (`title`).
* ### `questions.parquet`
    * **What it is:** The evaluation and training query suite (e.g., from datasets like MuSiQue or HotpotQA).
    * **What's inside:** It tracks specific questions (`query_text`), unique question identifiers (`question_id`), the structural complexity (`n_hops`), a list of the exact paragraphs that contain the true answers (`gold_node_ids`), and a pre-filtered sandbox pool of 20 paragraphs containing both the answers and noise (`candidate_node_ids`).

---

## 📂 2. The Artifacts Folder (`data/artifacts/`)
The **Artifacts** folder holds the heavy, compiled mathematical files. Humans cannot read these directly, but they are highly optimized for fast cloud computations and GPU processing.

* ### `doc_emb.npy`
    * **What it is:** The dense vector matrix of your master knowledge base.
    * **How it was made:** We blended the parent title and paragraph text together (`title + " - " + paragraph_text`) to eliminate context ambiguity, then passed it through the `BAAI/bge-base-en-v1.5` transformer on a GPU.
    * **Shape:** `(Number of Documents, 768)`. Every single paragraph is now represented as a highly compressed list of 768 numbers, unit-normalized to a mathematical length of exactly `1.0`.
* ### `query_emb.npy`
    * **What it is:** The dense vector matrix representing user questions.
    * **How it was made:** Encoded using the exact same BGE model, but prefixed with a search instruction (*"Represent this sentence for searching relevant passages: "*).
    * **Shape:** `(Number of Queries, 768)`.
* ### `query_ids.npy`
    * **What it is:** A simple index sequence of string IDs.
    * **Why it exists:** Dense matrices don't remember names or keys—they are just anonymous stacks of floating-point numbers. Row `15` of `query_emb.npy` is just vectors. This file ensures that Row `15` of `query_ids.npy` contains the exact question ID matching your original text table, letting you calculate your downstream metrics accurately.
* ### `embeddings_manifest.json`
    * **What it is:** A lightweight safety lock configuration file.
    * **Why it exists:** It records data boundaries (dimensions, counts, model names) and a deterministic hash of your raw corpus structure. When your local scripts initialize, they read this file first. If you accidentally alter your interim text tables later, this file throws a warning before you waste hours training an RL policy on mismatched row indices.

---