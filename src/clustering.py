"""K-means selection on the PCA representation."""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering


def cluster_stocks(components, n_clusters=None):
    max_k = min(6, len(components) - 1)
    if max_k < 2:
        raise ValueError("At least three stocks are needed for clustering.")
    scores, candidates = {}, range(2, max_k + 1)
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(components)
        scores[k] = {"silhouette": float(silhouette_score(components, labels)),
                     "davies_bouldin": float(davies_bouldin_score(components, labels))}
    if n_clusters is None:
        best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    else:
        if int(n_clusters) not in scores:
            raise ValueError(f"Cluster count must be between 2 and {max_k} for this stock sample.")
        best_k = int(n_clusters)
    model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = model.fit_predict(components)
    return labels, model, scores


def temporal_stability(features, components, labels, fraction=0.7):
    """Compare full-sample clusters with clusters fit on an early chronological return subperiod."""
    n = len(components)
    split = max(3, min(n - 1, int(n * fraction)))
    k = len(set(labels))
    if split <= k or n - split < 1:
        return {"adjusted_rand_index": None, "hierarchical_silhouette": None}
    early = KMeans(n_clusters=k, n_init=20, random_state=42).fit(components.iloc[:split])
    # ARI is computed for the overlapping early segment to avoid assigning unseen observations.
    ari = adjusted_rand_score(labels[:split], early.labels_)
    hierarchical = AgglomerativeClustering(n_clusters=k).fit_predict(components)
    score = float(silhouette_score(components, hierarchical)) if len(set(hierarchical)) > 1 else None
    return {"adjusted_rand_index": float(ari), "hierarchical_silhouette": score}
