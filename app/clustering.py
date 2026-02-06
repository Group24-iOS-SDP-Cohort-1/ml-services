import hdbscan
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from app.labeling import extract_keywords, generate_cluster_summary


def run_hdbscan(
    embeddings: list[list[float]],
    texts: list[str],
    min_cluster_size: int = 2,
    min_samples: int = 1
):
    # -------- Convert embeddings --------
    X = np.array(embeddings)

    # -------- Normalize embeddings --------
    X = normalize(X)

    # -------- PCA for stability --------
    if X.shape[0] >= 5:
        X = PCA(n_components=min(10, X.shape[1])).fit_transform(X)

    # -------- Run HDBSCAN --------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(X)

    # -------- Outliers --------
    outliers = [i for i, l in enumerate(labels) if l == -1]

    # -------- Cluster Sizes --------
    cluster_sizes = {}
    for label in labels:
        if label != -1:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    # -------- Cluster Grouping --------
    cluster_texts = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_texts.setdefault(label, []).append(idx)

    # -------- Unified Labels per Cluster --------
    unified_labels = {}

    for cluster_id, indices in cluster_texts.items():
        cluster_docs = [texts[i] for i in indices]

        keywords = extract_keywords(cluster_docs)

        title, desc = generate_cluster_summary(keywords)

        unified_labels[cluster_id] = {
            "title": title,
            "description": desc,
            "keywords": keywords,
            "examples": cluster_docs[:3]
        }

    # -------- Gap Analysis --------
    gaps = []

    if len(cluster_sizes) == 0:
        gaps.append("No meaningful clusters found. Content is too diverse.")

    if len(outliers) > len(labels) * 0.4:
        gaps.append("Too many outliers. Query may be too broad.")

    for cid, size in cluster_sizes.items():
        if size < 3:
            gaps.append(f"Cluster {cid} is weak (only {size} items).")

    # -------- Final Response --------
    return {
        "labels": labels.tolist(),
        "outliers": outliers,
        "clusters": cluster_sizes,
        "cluster_texts": cluster_texts,
        "unified_labels": unified_labels,
        "gaps": gaps
    }
