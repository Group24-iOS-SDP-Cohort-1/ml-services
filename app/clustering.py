import hdbscan
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def run_hdbscan(
    embeddings: list[list[float]],
    min_cluster_size: int = 2,
    min_samples: int = 1
):
    X = np.array(embeddings)

    # Normalize embeddings (CRITICAL for cosine-like behavior)
    X = normalize(X)

    # Dimensionality reduction for stability
    if X.shape[0] >= 5:
        X = PCA(n_components=min(10, X.shape[1])).fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(X)

    outliers = [i for i, l in enumerate(labels) if l == -1]

    cluster_sizes = {}
    for label in labels:
        if label != -1:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    gaps = [
        f"Low content volume in cluster {k}"
        for k, v in cluster_sizes.items()
        if v < min_cluster_size
    ]

    return {
        "labels": labels.tolist(),
        "outliers": outliers,
        "clusters": cluster_sizes,
        "gaps": gaps
    }
