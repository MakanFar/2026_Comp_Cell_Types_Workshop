"""
Baseline gene selection methods for comparison with the learned encoder.

Methods
-------
- ``pca``       Top genes by summed squared PCA loadings (same as Spapros core)
- ``de``        Top differentially expressed genes per cell type
- ``spapros``   Full Spapros ProbesetSelector pipeline
- ``random``    Random gene selection (with seed)
- ``hvg``       Top highly variable genes by dispersion (scRNA-seq only)

Important: for MERFISH data, all ~500 genes are already a curated panel —
there is no HVG concept.  Every method uses adata.var_names as its pool
directly.  ``select_hvg`` is disabled for MERFISH (falls back to PCA ranking).

Each function returns a list of gene names of length ``n``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc


# ─────────────────────────────────────────────────────────────────────────────
# PCA loading-based selection
# ─────────────────────────────────────────────────────────────────────────────

def _get_gene_pool(adata: sc.AnnData, genes_key: str = "highly_variable") -> sc.AnnData:
    """Return the appropriate gene pool for a dataset.

    For MERFISH data all genes are used directly.
    For scRNA-seq data the HVG mask is applied.
    """
    is_merfish = adata.uns.get("is_merfish", False)
    if is_merfish:
        return adata   # all ~500 genes — no filtering
    if genes_key in adata.var.columns:
        return adata[:, adata.var[genes_key]].copy()
    return adata


def select_pca(
    adata: sc.AnnData,
    n: int,
    n_pcs: int = 30,
    genes_key: str = "highly_variable",
    variance_scaled: bool = False,
) -> list[str]:
    """Select genes by summed squared PCA loadings.

    For MERFISH data, all ~500 genes are used as the pool (no HVG filtering).
    For scRNA-seq, the HVG subset is used.

    Args:
        adata:           Log-normalised AnnData.
        n:               Number of genes to select.
        n_pcs:           Number of principal components.
        genes_key:       HVG column name (ignored for MERFISH).
        variance_scaled: Scale scores by explained variance ratio.

    Returns:
        List of ``n`` gene names.
    """
    a = _get_gene_pool(adata, genes_key).copy()
    sc.pp.pca(a, n_comps=min(n_pcs, a.n_vars - 1))

    loadings = a.varm["PCs"]                          # (n_genes, n_pcs)
    if variance_scaled:
        var_ratio = a.uns["pca"]["variance_ratio"]    # (n_pcs,)
        scores = (loadings ** 2 * var_ratio).sum(axis=1)
    else:
        scores = (loadings ** 2).sum(axis=1)

    top_idx = np.argsort(scores)[::-1][:n]
    return list(a.var_names[top_idx])


# ─────────────────────────────────────────────────────────────────────────────
# DE-based selection
# ─────────────────────────────────────────────────────────────────────────────

def select_de(
    adata: sc.AnnData,
    n: int,
    celltype_key: str = "subclass",
    genes_key: str = "highly_variable",
    n_per_group: int = 3,
    method: str = "wilcoxon",
) -> list[str]:
    """Select the top DE genes across all cell types (1-vs-all).

    For MERFISH data, all ~500 genes are tested (no HVG filtering).
    For scRNA-seq, the HVG subset is used as the test pool.

    Args:
        adata:          Log-normalised AnnData with cell type annotations.
        n:              Total number of genes to select.
        celltype_key:   Column in ``adata.obs`` with cell type labels.
        n_per_group:    Top genes to take from each cell type initially.
        method:         DE test method (``"wilcoxon"`` or ``"t-test"``).

    Returns:
        List of up to ``n`` gene names.
    """
    a = _get_gene_pool(adata, genes_key).copy()

    sc.tl.rank_genes_groups(
        a,
        groupby=celltype_key,
        method=method,
        n_genes=a.n_vars,
        rankby_abs=False,
    )

    groups = a.uns["rank_genes_groups"]["names"].dtype.names
    selected: list[str] = []

    rank = 0
    while len(selected) < n and rank < a.n_vars:
        for group in groups:
            gene = str(a.uns["rank_genes_groups"]["names"][rank][groups.index(group)])
            if gene not in selected and gene in adata.var_names:
                selected.append(gene)
                if len(selected) >= n:
                    break
        rank += 1

    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Spapros full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def select_spapros(
    adata: sc.AnnData,
    n: int,
    celltype_key: str = "subclass",
    genes_key: str = "highly_variable",
    n_pca_genes: int = 200,
    n_jobs: int = -1,
    save_dir: str | None = None,
    verbosity: int = 1,
) -> list[str]:
    """Run the full Spapros ProbesetSelector pipeline.

    Requires spapros to be installed (``pip install spapros``).

    Returns:
        List of ``n`` selected gene names.
    """
    try:
        import spapros as sp
    except ImportError:
        raise ImportError(
            "spapros not installed. Run: pip install spapros\n"
            "Or use the local repo: pip install -e /path/to/spapros"
        )

    selector = sp.se.ProbesetSelector(
        adata,
        celltype_key=celltype_key,
        genes_key=genes_key,
        n=n,
        n_pca_genes=n_pca_genes,
        n_min_markers=2,
        save_dir=save_dir,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )
    selector.select_probeset()

    selected = selector.probeset[selector.probeset["selection"]].index.tolist()
    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Top highly variable genes
# ─────────────────────────────────────────────────────────────────────────────

def select_hvg(adata: sc.AnnData, n: int) -> list[str]:
    """Select top ``n`` genes by HVG dispersion score.

    Not meaningful for MERFISH data (all ~500 genes are already a curated
    targeted panel — there is no variance-based ranking to do).  For MERFISH
    this falls back to PCA-based selection to keep the comparison honest.
    """
    if adata.uns.get("is_merfish", False):
        print("  Note: HVG selection is not applicable to MERFISH data "
              "(already a targeted panel). Using PCA ranking instead.")
        return select_pca(adata, n)

    if "highly_variable_rank" in adata.var.columns:
        ranked = adata.var.sort_values("highly_variable_rank")
        return ranked.index[:n].tolist()
    elif "dispersions_norm" in adata.var.columns:
        ranked = adata.var.sort_values("dispersions_norm", ascending=False)
        return ranked.index[:n].tolist()
    else:
        a = adata.copy()
        sc.pp.highly_variable_genes(a, n_top_genes=n)
        return a.var_names[a.var["highly_variable"]].tolist()[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Random selection
# ─────────────────────────────────────────────────────────────────────────────

def select_random(
    adata: sc.AnnData,
    n: int,
    genes_key: str = "highly_variable",
    seed: int = 42,
) -> list[str]:
    """Random gene selection.

    For MERFISH: draws from all ~500 genes.
    For scRNA-seq: draws from the HVG pool.
    """
    rng = np.random.default_rng(seed)
    a = _get_gene_pool(adata, genes_key)
    pool = list(a.var_names)
    return list(rng.choice(pool, size=min(n, len(pool)), replace=False))


# ─────────────────────────────────────────────────────────────────────────────
# Run all baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(
    adata: sc.AnnData,
    panel_sizes: list[int],
    celltype_key: str = "subclass",
    genes_key: str = "highly_variable",
    include_spapros: bool = True,
    spapros_save_dir: str | None = None,
    random_seeds: list[int] | None = None,
    n_jobs: int = -1,
) -> dict[str, dict[int, list[str]]]:
    """Run all baseline selectors at all panel sizes.

    Args:
        adata:          Preprocessed AnnData.
        panel_sizes:    List of k values (e.g. [10, 25, 50, 100]).
        include_spapros: Run Spapros (slower but most comparable).
        random_seeds:   Seeds for random baselines.

    Returns:
        Nested dict: ``{method_name: {panel_size: gene_list}}``.
    """
    if random_seeds is None:
        random_seeds = [0, 1, 2]

    results: dict[str, dict[int, list[str]]] = {}

    for n in panel_sizes:
        print(f"\n── Panel size: {n} ──────────────────────")

        # PCA
        print("  PCA selection ...")
        results.setdefault("pca", {})[n] = select_pca(adata, n, genes_key=genes_key)

        # DE
        print("  DE selection ...")
        results.setdefault("de", {})[n] = select_de(adata, n, celltype_key=celltype_key, genes_key=genes_key)

        # HVG
        print("  HVG selection ...")
        results.setdefault("hvg", {})[n] = select_hvg(adata, n)

        # Random (multiple seeds → report mean later)
        for seed in random_seeds:
            key = f"random_seed{seed}"
            results.setdefault(key, {})[n] = select_random(adata, n, genes_key=genes_key, seed=seed)

        # Spapros
        if include_spapros:
            print("  Spapros selection ...")
            save = f"{spapros_save_dir}/spapros_n{n}/" if spapros_save_dir else None
            results.setdefault("spapros", {})[n] = select_spapros(
                adata, n,
                celltype_key=celltype_key,
                genes_key=genes_key,
                save_dir=save,
                n_jobs=n_jobs,
            )

    return results


def baselines_to_dataframe(
    baselines: dict[str, dict[int, list[str]]],
) -> pd.DataFrame:
    """Convert nested baselines dict to a long-format DataFrame for easy analysis."""
    rows = []
    for method, size_dict in baselines.items():
        for n, genes in size_dict.items():
            rows.append({"method": method, "n_genes": n, "genes": genes})
    return pd.DataFrame(rows)
