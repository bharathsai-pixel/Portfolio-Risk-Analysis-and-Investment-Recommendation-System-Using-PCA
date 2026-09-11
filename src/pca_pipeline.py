"""PCA fitting and factor-loading helpers."""
import pandas as pd
from sklearn.decomposition import PCA


def run_pca(scaled_features, variance_target=0.90):
    pca = PCA(n_components=variance_target, svd_solver="full", random_state=42)
    components = pca.fit_transform(scaled_features)
    columns = [f"PC{i}" for i in range(1, components.shape[1] + 1)]
    return pca, pd.DataFrame(components, index=scaled_features.index, columns=columns)


def factor_loadings(pca, feature_names):
    columns = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    return pd.DataFrame(pca.components_.T, index=feature_names, columns=columns)
