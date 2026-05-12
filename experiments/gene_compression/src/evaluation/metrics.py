"""
Evaluation metrics for gene panel compression.

Metrics
-------
1. Reconstruction quality
   - MSE / Pearson-R per gene, per cell, overall
   - Top-expressed gene recovery fraction

2. Cell type recovery
   - XGBoost classifier F1 (macro + per cell type)
   - k-NN cell type label accuracy from latent space

3. Transcriptional structure preservation
   - Clustering NMI vs. full transcriptome (mirrors Spapros cluster_similarity)
   - k-NN graph overlap (mirrors Spapros knn_overlap)

4. Spatial organization (for MERFISH data)
   - Moran's I for key marker genes in reconstructed vs. original
   - Spatial niche composition similarity

All functions accept either a gene panel (list of gene names) with the
full AnnData, or pre-computed latent embeddings.  Compatible with the
Spapros ProbesetEvaluator where noted.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstruction quality
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_mse(
    x_orig: np.ndarray,
    x_recon: np.ndarray,
) -> dict[str, float]:
    """Per-cell and per-gene MSE and overall Pearson R."""
    mse_per_cell = np.mean((x_orig - x_recon) ** 2, axis=1)
    mse_per_gene = np.mean((x_orig - x_recon) ** 2, axis=0)

    # Flatten for overall Pearson
    flat_orig = x_orig.flatten()
    flat_recon = x_recon.flatten()
    # Sample for speed on large arrays
    if len(flat_orig) > 500_000:
        idx = np.random.choice(len(flat_orig), 500_000, replace=False)
        flat_orig, flat_recon = flat_orig[idx], flat_recon[idx]
    r, _ = pearsonr(flat_orig, flat_recon)

    return {
        "mse_mean": float(np.mean(mse_per_cell)),
        "mse_median": float(np.median(mse_per_cell)),
        "pearson_r": float(r),
        "mse_per_gene_mean": float(np.mean(mse_per_gene)),
    }


def gene_recovery_rate(
    selected_genes: list[str],
    reference_genes: list[str],
    top_k: int = 50,
) -> float:
    """Fraction of ``reference_genes[:top_k]`` that appear in ``selected_genes``."""
    top_ref = set(reference_genes[:top_k])
    return len(top_ref & set(selected_genes)) / len(top_ref)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cell type recovery
# ─────────────────────────────────────────────────────────────────────────────

def celltype_f1_from_panel(
    adata: sc.AnnData,
    selected_genes: list[str],
    celltype_key: str = "ct_label",
    cv: int = 5,
    n_estimators: int = 200,
) -> dict[str, float]:
    """Train XGBoost on the selected gene panel and measure cell type F1.

    Uses cross-validation so no separate test set is required.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

    genes = [g for g in selected_genes if g in adata.var_names]
    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs[celltype_key].values

    clf = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False,
                        eval_metric="mlogloss", n_jobs=-1, verbosity=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return {"f1_macro": float(scores.mean()), "f1_std": float(scores.std())}


def celltype_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
    cv: int = 5,
) -> float:
    """k-NN cell type accuracy in the latent embedding space."""
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transcriptional structure preservation
# ─────────────────────────────────────────────────────────────────────────────

def clustering_nmi(
    adata_full: sc.AnnData,
    adata_panel: sc.AnnData,
    resolutions: list[float] | None = None,
    n_pcs: int = 30,
    n_neighbors: int = 15,
) -> dict[str, float]:
    """NMI between full-transcriptome and panel Leiden clusterings.

    Runs Leiden at multiple resolutions and averages NMI.
    Mirrors the Spapros ``cluster_similarity`` metric.

    Args:
        adata_full:   AnnData with full HVG expression (already log-normalised).
        adata_panel:  AnnData subset to selected genes.
        resolutions:  List of Leiden resolution values.

    Returns:
        Dict with mean NMI and per-resolution values.
    """
    if resolutions is None:
        resolutions = np.linspace(0.1, 2.0, 15).tolist()

    def _cluster(adata: sc.AnnData, res: float, suffix: str) -> np.ndarray:
        a = adata.copy()
        sc.pp.pca(a, n_comps=min(n_pcs, a.n_vars - 1))
        sc.pp.neighbors(a, n_neighbors=n_neighbors, n_pcs=min(n_pcs, a.n_vars - 1))
        sc.tl.leiden(a, resolution=res, key_added=f"leiden_{suffix}")
        return a.obs[f"leiden_{suffix}"].values

    nmis = []
    for res in resolutions:
        labels_full = _cluster(adata_full, res, "full")
        labels_panel = _cluster(adata_panel, res, "panel")
        nmis.append(normalized_mutual_info_score(labels_full, labels_panel))

    result = {f"nmi_res_{r:.2f}": float(v) for r, v in zip(resolutions, nmis)}
    result["nmi_mean"] = float(np.mean(nmis))
    result["nmi_auc"] = float(np.trapz(nmis, resolutions) / (max(resolutions) - min(resolutions)))
    return result


def knn_overlap(
    adata_full: sc.AnnData,
    adata_panel: sc.AnnData,
    ks: list[int] | None = None,
    n_pcs: int = 30,
    n_subsample: int = 2000,
) -> dict[str, float]:
    """Mean overlap of k-NN graphs between full and panel expression.

    Mirrors the Spapros ``knn_overlap`` metric.

    Args:
        adata_full:    AnnData with full HVG expression.
        adata_panel:   AnnData subset to selected genes.
        ks:            List of k values to evaluate.
        n_subsample:   Subsample cells for speed.

    Returns:
        Dict with mean overlap per k and overall mean.
    """
    if ks is None:
        ks = [5, 10, 15, 20, 30]

    # Subsample for speed
    n = min(n_subsample, adata_full.n_obs)
    idx = np.random.choice(adata_full.n_obs, n, replace=False)

    def _pca_coords(adata: sc.AnnData, n_pcs: int) -> np.ndarray:
        a = adata.copy()
        sc.pp.pca(a, n_comps=min(n_pcs, a.n_vars - 1))
        return a.obsm["X_pca"]

    emb_full = _pca_coords(adata_full[idx], n_pcs)
    emb_panel = _pca_coords(adata_panel[idx], n_pcs)

    dist_full = cdist(emb_full, emb_full, metric="euclidean")
    dist_panel = cdist(emb_panel, emb_panel, metric="euclidean")

    results: dict[str, float] = {}
    for k in ks:
        nn_full = np.argsort(dist_full, axis=1)[:, 1: k + 1]
        nn_panel = np.argsort(dist_panel, axis=1)[:, 1: k + 1]
        overlaps = [
            len(set(nn_full[i]) & set(nn_panel[i])) / k
            for i in range(n)
        ]
        results[f"knn_overlap_k{k}"] = float(np.mean(overlaps))

    results["knn_overlap_mean"] = float(np.mean(list(results.values())))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spatial organization
# ─────────────────────────────────────────────────────────────────────────────

def morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 6,
) -> float:
    """Compute Moran's I spatial autocorrelation statistic.

    Higher values (close to 1) indicate stronger spatial clustering of the
    given variable.

    Args:
        values:      (n_cells,) expression or reconstruction values.
        coords:      (n_cells, 2 or 3) spatial coordinates.
        n_neighbors: Number of spatial neighbours for weight matrix.

    Returns:
        Moran's I scalar.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Binary spatial weights (inverse distance)
    n = len(values)
    W = np.zeros((n, n))
    for i in range(n):
        for j, d in zip(indices[i, 1:], distances[i, 1:]):
            W[i, j] = 1.0 / (d + 1e-9)

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    z = values - values.mean()
    numerator = n * (z @ W @ z)
    denominator = (z ** 2).sum() * W.sum()
    return float(numerator / denominator) if denominator != 0 else 0.0


def spatial_morans_comparison(
    adata_spatial: sc.AnnData,
    x_recon: np.ndarray,
    genes: list[str],
    n_neighbors: int = 6,
) -> pd.DataFrame:
    """Compare Moran's I between original and reconstructed expression.

    Args:
        adata_spatial:  AnnData with spatial coords in ``obsm["spatial"]``.
        x_recon:        (n_cells, n_genes) reconstructed expression.
        genes:          Gene names corresponding to columns of x_recon.

    Returns:
        DataFrame with columns [gene, morans_original, morans_recon, ratio].
    """
    coords = adata_spatial.obsm["spatial"][:, :2]   # use x, y only
    X_orig = adata_spatial.X
    if hasattr(X_orig, "toarray"):
        X_orig = X_orig.toarray()

    records = []
    for i, gene in enumerate(genes):
        if gene not in adata_spatial.var_names:
            continue
        g_idx = list(adata_spatial.var_names).index(gene)
        mi_orig = morans_i(X_orig[:, g_idx], coords, n_neighbors)
        mi_recon = morans_i(x_recon[:, i], coords, n_neighbors)
        records.append({
            "gene": gene,
            "morans_original": mi_orig,
            "morans_recon": mi_recon,
            "ratio": mi_recon / (mi_orig + 1e-9),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_panel(
    adata: sc.AnnData,
    selected_genes: list[str],
    embeddings: np.ndarray | None = None,
    celltype_key: str = "ct_label",
    run_clustering: bool = True,
    run_knn: bool = True,
    run_ct_f1: bool = True,
) -> dict[str, float | dict]:
    """Run the full evaluation suite for a given gene panel.

    Args:
        adata:          Full AnnData (log-normalised, HVG computed).
        selected_genes: List of selected gene names.
        embeddings:     (n_cells, latent_dim) latent embeddings from the
                        encoder.  If None, the selected gene expression is
                        used directly for structure metrics.
        celltype_key:   obs column for integer cell type labels.
        run_clustering: Run the clustering NMI metric.
        run_knn:        Run the k-NN overlap metric.
        run_ct_f1:      Run the XGBoost cell type F1 metric.

    Returns:
        Flat dict of metric_name → value.
    """
    results: dict[str, float | dict] = {"n_genes": len(selected_genes)}

    genes_in_adata = [g for g in selected_genes if g in adata.var_names]
    adata_panel = adata[:, genes_in_adata].copy()
    adata_full = adata[:, adata.var["highly_variable"]].copy() if "highly_variable" in adata.var.columns else adata.copy()

    if run_ct_f1:
        f1_res = celltype_f1_from_panel(adata, genes_in_adata, celltype_key)
        results.update(f1_res)

    if embeddings is not None:
        labels = adata.obs[celltype_key].values
        results["knn_ct_accuracy"] = celltype_knn_accuracy(embeddings, labels)

    if run_clustering:
        nmi_res = clustering_nmi(adata_full, adata_panel)
        results.update(nmi_res)

    if run_knn:
        knn_res = knn_overlap(adata_full, adata_panel)
        results.update(knn_res)

    return results


def compare_panels(
    adata: sc.AnnData,
    panels: dict[str, list[str]],
    celltype_key: str = "ct_label",
    **eval_kwargs,
) -> pd.DataFrame:
    """Evaluate and compare multiple gene panels.

    Args:
        adata:   Full AnnData.
        panels:  Dict of {method_name: gene_list}.

    Returns:
        DataFrame with methods as rows and metrics as columns.
    """
    rows = []
    for name, genes in panels.items():
        print(f"Evaluating: {name} ({len(genes)} genes) ...")
        res = evaluate_panel(adata, genes, celltype_key=celltype_key, **eval_kwargs)
        res["method"] = name
        rows.append(res)

    df = pd.DataFrame(rows).set_index("method")
    return df
