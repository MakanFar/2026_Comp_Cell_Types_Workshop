"""
Visualization utilities for gene panel compression experiments.

Key plots:
  - UMAP comparison: full transcriptome vs. panel vs. latent space
  - Reconstruction scatter: original vs. reconstructed expression
  - Panel size comparison bar chart
  - Spatial expression maps: original vs. reconstructed
  - Selection overlap (UpSet / Venn across methods)
  - Temperature annealing curve
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


# ─────────────────────────────────────────────────────────────────────────────
# Color palette consistent with ABC Atlas publications
# ─────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "concrete_ae": "#2196F3",   # blue
    "stg_ae":      "#00BCD4",   # cyan
    "spapros":     "#4CAF50",   # green
    "pca":         "#FF9800",   # orange
    "de":          "#9C27B0",   # purple
    "hvg":         "#795548",   # brown
    "random":      "#9E9E9E",   # grey
}


# ─────────────────────────────────────────────────────────────────────────────
# UMAP comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_comparison(
    adata_full: sc.AnnData,
    adata_panel: sc.AnnData,
    embeddings: np.ndarray | None,
    celltype_key: str = "subclass",
    n_pcs: int = 30,
    figsize: tuple[float, float] = (18, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """Three-panel UMAP: full transcriptome | panel | learned latent space.

    Args:
        adata_full:   Full HVG AnnData.
        adata_panel:  AnnData subset to selected genes.
        embeddings:   (n_cells, latent_dim) optional latent embeddings.
        celltype_key: Obs key for coloring.
    """
    n_panels = 3 if embeddings is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    def _compute_umap(a: sc.AnnData, title: str, ax: plt.Axes) -> None:
        a = a.copy()
        sc.pp.pca(a, n_comps=min(n_pcs, a.n_vars - 1))
        sc.pp.neighbors(a, n_pcs=min(n_pcs, a.n_vars - 1))
        sc.tl.umap(a)
        sc.pl.umap(a, color=celltype_key, ax=ax, show=False,
                   legend_loc="none", title=title, frameon=False)

    _compute_umap(adata_full, "Full transcriptome", axes[0])
    _compute_umap(adata_panel, f"Panel ({adata_panel.n_vars} genes)", axes[1])

    if embeddings is not None:
        # Add embeddings to a copy of adata for UMAP
        a_lat = adata_full.copy()
        a_lat.obsm["X_latent"] = embeddings
        # Compute UMAP from latent embeddings
        sc.pp.neighbors(a_lat, use_rep="X_latent")
        sc.tl.umap(a_lat)
        sc.pl.umap(a_lat, color=celltype_key, ax=axes[2], show=False,
                   legend_loc="right margin", title="Learned latent space", frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_panel_comparison(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
    panel_sizes: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing methods across panel sizes for multiple metrics.

    Args:
        results_df:   DataFrame with columns: method, n_genes, metric_1, metric_2, ...
        metrics:      List of metric column names to plot.
        panel_sizes:  Subset of panel sizes to show.
    """
    if metrics is None:
        metrics = [c for c in results_df.columns if c not in ["method", "n_genes"]]
    if panel_sizes is not None:
        results_df = results_df[results_df["n_genes"].isin(panel_sizes)]

    methods = results_df["method"].unique()
    sizes = sorted(results_df["n_genes"].unique())

    n_rows = len(metrics)
    n_cols = len(sizes)
    fig_h = figsize[1] if figsize else 4 * n_rows
    fig_w = figsize[0] if figsize else 5 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for row, metric in enumerate(metrics):
        for col, size in enumerate(sizes):
            ax = axes[row][col]
            sub = results_df[results_df["n_genes"] == size]
            values = [sub[sub["method"] == m][metric].values[0] if len(sub[sub["method"] == m]) > 0 else 0 for m in methods]
            colors = [METHOD_COLORS.get(m, "#607D8B") for m in methods]
            bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=0.8)

            ax.set_title(f"{metric}  |  n={size}", fontsize=9)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Gene Panel Compression — Method Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_scatter(
    x_orig: np.ndarray,
    x_recon: np.ndarray,
    n_genes_to_show: int = 6,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] = (14, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter plots: original vs. reconstructed expression for selected genes."""
    n_cells_sample = min(2000, x_orig.shape[0])
    idx = np.random.choice(x_orig.shape[0], n_cells_sample, replace=False)

    # Show genes with highest variance (most interesting)
    gene_var = x_orig.var(axis=0)
    top_genes = np.argsort(gene_var)[::-1][:n_genes_to_show]

    fig, axes = plt.subplots(2, n_genes_to_show // 2, figsize=figsize)
    axes = axes.flatten()

    for i, g_idx in enumerate(top_genes):
        ax = axes[i]
        ax.scatter(x_orig[idx, g_idx], x_recon[idx, g_idx],
                   alpha=0.3, s=5, color="#2196F3", rasterized=True)
        ax.plot([0, x_orig[:, g_idx].max()], [0, x_orig[:, g_idx].max()],
                "r--", linewidth=1, alpha=0.7)

        r = np.corrcoef(x_orig[idx, g_idx], x_recon[idx, g_idx])[0, 1]
        title = var_names[g_idx] if var_names else f"Gene {g_idx}"
        ax.set_title(f"{title}\nPearson r={r:.3f}", fontsize=8)
        ax.set_xlabel("Original", fontsize=7)
        ax.set_ylabel("Reconstructed", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Reconstruction Quality — Top Variable Genes", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Spatial expression comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_comparison(
    adata_spatial: sc.AnnData,
    x_recon: np.ndarray,
    genes: list[str],
    n_genes_show: int = 4,
    section: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Side-by-side spatial expression maps: original vs. reconstructed.

    Args:
        adata_spatial:  AnnData with ``obsm["spatial"]`` coordinates.
        x_recon:        (n_cells, n_genes) reconstructed expression.
        genes:          Gene names for columns of x_recon.
        section:        Z coordinate to slice (if 3D).  None = use all cells.
    """
    coords = adata_spatial.obsm["spatial"]

    if section is not None and coords.shape[1] >= 3:
        z_vals = coords[:, 2]
        mask = np.abs(z_vals - section) < (np.ptp(z_vals) / 50)
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(adata_spatial))

    xy = coords[idx, :2]
    X_orig = adata_spatial.X
    if hasattr(X_orig, "toarray"):
        X_orig = X_orig.toarray()

    show_genes = genes[:n_genes_show]
    fig, axes = plt.subplots(2, len(show_genes), figsize=(4 * len(show_genes), 8))

    for col, gene in enumerate(show_genes):
        if gene not in adata_spatial.var_names:
            continue
        g_idx_adata = list(adata_spatial.var_names).index(gene)
        g_idx_recon = genes.index(gene)

        orig_vals = X_orig[idx, g_idx_adata]
        recon_vals = x_recon[idx, g_idx_recon]
        vmax = np.percentile(orig_vals, 99)

        for row, (vals, title) in enumerate([
            (orig_vals, f"{gene}\n(original)"),
            (recon_vals, f"{gene}\n(reconstructed)"),
        ]):
            ax = axes[row][col]
            sc_plot = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="magma",
                                 s=1, vmin=0, vmax=vmax, rasterized=True)
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")
            plt.colorbar(sc_plot, ax=ax, shrink=0.6)

    plt.suptitle("Spatial Expression: Original vs. Reconstructed", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: list[dict],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training and validation losses + temperature annealing curve."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Total loss
    axes[0].plot(df["epoch"], df["train_total"], label="train", color="#2196F3")
    axes[0].plot(df["epoch"], df["val_total"], label="val", color="#F44336")
    axes[0].set_title("Total Loss")
    axes[0].legend(); axes[0].set_xlabel("Epoch")

    # Component losses
    for col in ["train_recon", "train_ct"]:
        if col in df.columns:
            axes[1].plot(df["epoch"], df[col], label=col.replace("train_", ""))
    axes[1].set_title("Loss Components (train)")
    axes[1].legend(); axes[1].set_xlabel("Epoch")

    # Temperature
    axes[2].plot(df["epoch"], df["temperature"], color="#FF9800", linewidth=2)
    axes[2].set_title("Concrete Temperature τ")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("τ")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
