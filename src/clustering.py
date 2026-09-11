"""K-means selection on the PCA representation."""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_stocks(components):
    max_k = min(6, len(components) - 1)
    scores, candidates = {}, range(2, max_k + 1)
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(components)
        scores[k] = float(silhouette_score(components, labels))
    best_k = max(scores, key=scores.get)
    model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    return model.fit_predict(components), model, scores
