import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import plotly.express as px
import joblib

# ==========================================
# 1. Baseline Metrics
# ==========================================
def get_average_random_pair_similarity(embeddings, num_pairs=1000, seed=42):
    """Calculates the average cosine similarity between random pairs to establish a baseline."""
    rng = np.random.default_rng(seed)
    num_docs = len(embeddings)
    
    idx1 = rng.choice(num_docs, size=num_pairs, replace=True)
    idx2 = rng.choice(num_docs, size=num_pairs, replace=True)
    
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    
    dot_products = np.sum(emb1 * emb2, axis=1)
    norms1 = np.linalg.norm(emb1, axis=1)
    norms2 = np.linalg.norm(emb2, axis=1)
    
    similarities = dot_products / np.maximum(norms1 * norms2, 1e-10)
    
    return float(np.mean(similarities))


# ==========================================
# 2. Evaluation Helpers
# ==========================================
def get_cluster_metrics_table(embeddings, labels, cluster_level_name="Coarse", sample=500, seed=42):
    """Generates the metrics table (Intra, Inter, Separation) for a given set of labels."""
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    k = len(unique)
    
    centroids = np.array([embeddings[labels == c].mean(axis=0) for c in unique])
    centroids = normalize(centroids, norm='l2')

    results = []
    for i, c in enumerate(unique):
        pts = embeddings[labels == c]
        n = min(sample, len(pts))
        if n == 0: continue
            
        idx = rng.choice(len(pts), size=n, replace=False)
        sub = pts[idx]

        intra_sim = float((sub @ centroids[i]).mean())
        if k > 1:
            other = np.delete(centroids, i, axis=0)
            inter_sim = float((sub @ other.T).max(axis=1).mean())
        else:
            inter_sim = 0.0
            
        results.append({
            "Level": cluster_level_name,
            "Cluster ID": c,
            "Size": len(pts),
            "Intra-Sim": intra_sim,
            "Inter-Sim": inter_sim,
            "Separation": intra_sim - inter_sim
        })
        
    return pd.DataFrame(results)

def _score_clustering(embeddings, labels, sample=500, seed=42):
    """Helper to calculate average separation to determine the 'best' Fine K."""
    df = get_cluster_metrics_table(embeddings, labels, sample=sample, seed=seed)
    return df["Separation"].mean()

def find_best_fine_kmeans(embeddings, k_options, seed=42):
    """Iterates through k_options, fits KMeans, and returns the fine model with the best Separation score."""
    best_k = None
    best_score = -float('inf')
    best_model = None
    best_labels = None

    for k in k_options:
        model = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        labels = model.fit_predict(embeddings)
        score = _score_clustering(embeddings, labels, seed=seed)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels
            
    return best_k, best_model, best_labels


# ==========================================
# 3. Draft & Execution Core
# ==========================================
def check_coarse_elbow(embeddings, k_options, sample_size=50000, seed=42):
    """Subsamples the dataset and calculates KMeans inertia across k_options."""
    rng = np.random.default_rng(seed)
    n_docs = len(embeddings)
    
    if n_docs > sample_size:
        print(f"Subsampling {sample_size:,} docs out of {n_docs:,} for Elbow check...")
        idx = rng.choice(n_docs, size=sample_size, replace=False)
        sub_embs = embeddings[idx]
    else:
        print(f"Using full dataset ({n_docs:,} docs) for Elbow check.")
        sub_embs = embeddings

    sub_embs_norm = normalize(sub_embs, norm='l2')
    
    inertias = []
    print("Calculating inertias...")
    for k in k_options:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        kmeans.fit(sub_embs_norm)
        inertias.append(kmeans.inertia_)
        print(f"  k={k} -> Inertia: {kmeans.inertia_:.4f}")
        
    return pd.DataFrame({"k": k_options, "Inertia": inertias})


def build_dynamic_hierarchy(doc_emb, chosen_coarse_k, fine_k_options, seed=42):
    """Runs the full dataset using chosen coarse_k, then optimizes fine clusters."""
    print(f"L2 Normalizing complete document dataset...")
    docs_norm = normalize(doc_emb, norm='l2')

    print(f"Fitting full dataset to Coarse K={chosen_coarse_k}...")
    coarse_model = KMeans(n_clusters=chosen_coarse_k, random_state=seed, n_init=10)
    coarse_labels = coarse_model.fit_predict(docs_norm)

    fine_models = {} 
    hierarchical_labels = np.empty(len(docs_norm), dtype=object)
    
    print(f"\nOptimizing Fine K for each of the {chosen_coarse_k} coarse clusters...")
    for coarse_id in range(chosen_coarse_k):
        idx = np.where(coarse_labels == coarse_id)[0]
        sub_embs = docs_norm[idx]
        
        valid_fine_options = [k for k in fine_k_options if k < len(sub_embs)]
        if not valid_fine_options:
            valid_fine_options = [max(1, len(sub_embs) - 1)]

        _, fine_model, fine_labels = find_best_fine_kmeans(sub_embs, valid_fine_options, seed)
        fine_models[coarse_id] = fine_model
        
        for doc_i, f_label in zip(idx, fine_labels):
            hierarchical_labels[doc_i] = f"C{coarse_id}_F{f_label}"

    df_coarse = get_cluster_metrics_table(docs_norm, coarse_labels, "Coarse")
    df_fine = get_cluster_metrics_table(docs_norm, hierarchical_labels, "Fine")
    df_all = pd.concat([df_coarse, df_fine], ignore_index=True)

    print("\nFull Hierarchy Build Complete!")
    return {
        "coarse_model": coarse_model,
        "fine_models": fine_models,
        "doc_coarse_labels": coarse_labels,
        "doc_hierarchical_labels": hierarchical_labels,
        "metrics_df": df_all
    }


def apply_hierarchy_to_queries(query_emb, coarse_model, fine_models):
    """Applies a pre-trained hierarchy structure to target query embeddings."""
    queries_norm = normalize(query_emb, norm='l2')
    query_coarse_labels = coarse_model.predict(queries_norm)
    query_hierarchical_labels = np.empty(len(queries_norm), dtype=object)

    for coarse_id, f_model in fine_models.items():
        idx = np.where(query_coarse_labels == coarse_id)[0]
        if len(idx) == 0: continue
            
        sub_queries = queries_norm[idx]
        fine_labels = f_model.predict(sub_queries)
        
        for q_i, f_label in zip(idx, fine_labels):
            query_hierarchical_labels[q_i] = f"C{coarse_id}_F{f_label}"

    return query_coarse_labels, query_hierarchical_labels


# ==========================================
# 4. Plotting & Artifacts
# ==========================================
def plot_elbow_curve(elbow_data):
    """Plots the Elbow curve interactively."""
    fig = px.line(
        elbow_data, 
        x='k', y='Inertia', markers=True,
        title="Elbow Method: Choose your Coarse K",
        labels={'k': 'Number of Clusters (k)', 'Inertia': 'Inertia (Lower is tighter)'}
    )
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=elbow_data['k'].min(), dtick=10))
    return fig


def plot_hierarchy_tree(df_metrics):
    """Creates a rectangular tree plot (dendrogram style) showing hierarchy and sizes."""
    df_plot = df_metrics[df_metrics["Level"] == "Fine"].copy()
    
    df_plot["Coarse Parent"] = df_plot["Cluster ID"].apply(lambda x: "Coarse " + x.split("_")[0].replace("C", ""))
    df_plot["Fine Child"] = df_plot["Cluster ID"].apply(lambda x: "Fine " + x.split("_")[1].replace("F", ""))
    df_plot["Root"] = "All Documents"

    fig = px.icicle(
        df_plot,
        path=['Root', 'Coarse Parent', 'Fine Child'],
        values='Size',
        color='Separation',
        color_continuous_scale='RdYlGn',
        title="Hierarchical Clusters (Width = Size, Color = Separation)"
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig


def save_artifacts(save_dir, doc_labels, query_labels, coarse_model, fine_models):
    """Persists labels to .npy and trained models to compressed serialization files."""
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "doc_hierarchical_labels.npy"), doc_labels)
    np.save(os.path.join(save_dir, "query_hierarchical_labels.npy"), query_labels)
    
    joblib.dump(coarse_model, os.path.join(save_dir, "kmeans_coarse.pkl"))
    joblib.dump(fine_models, os.path.join(save_dir, "kmeans_fine_dict.pkl"))
    print(f"All artifacts successfully saved to {save_dir}/")


def format_hierarchical_metrics(df_metrics):
    """
    Takes the raw metrics DataFrame, groups Coarse and Fine clusters hierarchically,
    and formats the similarity scores as percentages with 2 decimal places.
    """
    df = df_metrics.copy()
    
    # 1. Determine the Coarse Parent for every row to enable grouping
    def get_parent(row):
        if row['Level'] == 'Coarse':
            return f"Coarse {row['Cluster ID']}"
        else:
            # Parses a string like "C0_F1" to extract just the "0"
            parent_num = str(row['Cluster ID']).split('_')[0].replace('C', '')
            return f"Coarse {parent_num}"
            
    df['Parent Group'] = df.apply(get_parent, axis=1)
    
    # 2. Sort so the Coarse parent is always at the top, followed by its Fine children
    df['_sort_level'] = df['Level'].map({'Coarse': 0, 'Fine': 1})
    df = df.sort_values(by=['Parent Group', '_sort_level', 'Cluster ID'])
    
    # 3. Create a Hierarchical MultiIndex
    df = df.set_index(['Parent Group', 'Level', 'Cluster ID']).drop(columns=['_sort_level'])
    
    # 4. Format the metric columns to percentages with 2 decimal places (e.g., 65.20%)
    format_dict = {
        'Intra-Sim': '{:.1%}',
        'Inter-Sim': '{:.1%}',
        'Separation': '{:.1%}'
    }
    
    # Return a Pandas Styler object which renders beautifully in Jupyter
    return df.style.format(format_dict)