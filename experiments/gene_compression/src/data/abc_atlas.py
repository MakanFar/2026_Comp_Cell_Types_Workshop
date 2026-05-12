"""
Data loading utilities for the Allen Brain Cell (ABC) Atlas.

Two distinct data modalities are handled here with different preprocessing:

  scRNA-seq (10X Chromium)
  ─────────────────────────
  ~32,000 genes per cell.  HVG selection to ~3000 genes is applied before
  feeding into the model.  Use case: learning a gene panel *from scratch*
  using the full transcriptome as reference.

  MERFISH whole mouse brain (Zhuang et al.)
  ──────────────────────────────────────────
  ~500 genes per cell — already a targeted spatial panel.  NO HVG selection:
  we use all genes directly.  The model's task is to find the most informative
  *subset* of those 500 genes (e.g. compress 500 → 10 / 25 / 50 / 100).
  Cells also carry (x, y, z) spatial coordinates.

Code Ocean paths
----------------
    /data/abc_atlas/expression_matrices/WMB-10Xv3/<date>/   — scRNA-seq h5ads
    /data/abc_atlas/expression_matrices/MERFISH-C57BL6J-638850/<date>/  — MERFISH h5ad
    /data/abc_atlas/metadata/WMB-10X/<date>/cell_metadata_with_cluster_annotation.csv

Set env var ABC_ATLAS_PATH to override the root.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ─────────────────────────────────────────────────────────────────────────────
# Path registry
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATA_ROOT = Path(
    os.environ.get("ABC_ATLAS_PATH", "/data/abc_atlas")
)

SCRNA_FILES = {
    "WMB-10Xv3-TH":              "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-TH-raw.h5ad",
    "WMB-10Xv3-CTXsp":           "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-CTXsp-raw.h5ad",
    "WMB-10Xv3-OLF":             "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-OLF-raw.h5ad",
    "WMB-10Xv3-HPF":             "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-HPF-raw.h5ad",
    "WMB-10Xv3-MB":              "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-MB-raw.h5ad",
    "WMB-10Xv3-HY":              "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-HY-raw.h5ad",
    "WMB-10Xv3-CB":              "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-CB-raw.h5ad",
    "WMB-10Xv3-non-neuronal":    "expression_matrices/WMB-10Xv3/20231215/WMB-10Xv3-non-neuronal-raw.h5ad",
}

# MERFISH whole mouse brain — one file, all brain regions, all cell types
MERFISH_FILES = {
    # ABC tutorial convention: abc_cache.get_file_path("MERFISH-C57BL6J-638850", "C57BL6J-638850/log2")
    "MERFISH-C57BL6J-638850": "expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad",
}

CELL_METADATA_FILE = "metadata/WMB-10X/20231215/cell_metadata_with_cluster_annotation.csv"
MERFISH_METADATA_FILE = "metadata/MERFISH-C57BL6J-638850/20230830/cell_metadata_with_cluster_annotation.csv"


# ─────────────────────────────────────────────────────────────────────────────
# scRNA-seq loader  (HVG selection applied)
# ─────────────────────────────────────────────────────────────────────────────

def load_abc_scrna(
    region: str = "WMB-10Xv3-TH",
    data_root: str | Path | None = None,
    celltype_key: str = "subclass",
    n_hvg: int = 3000,
    max_cells: int | None = None,
    random_state: int = 0,
) -> sc.AnnData:
    """Load and preprocess an ABC Atlas scRNA-seq region.

    HVG selection IS applied (32k → n_hvg genes).  Use this when designing a
    gene panel from scratch using the full transcriptome as reference.

    Args:
        region:        Key in SCRNA_FILES, or a direct path to an h5ad file.
        data_root:     Root directory (defaults to ABC_ATLAS_PATH env var).
        celltype_key:  Column in obs for cell type labels.
                       Options: "class", "subclass", "supertype", "cluster".
        n_hvg:         Number of highly variable genes to retain.
        max_cells:     Optional cell subsampling limit.
        random_state:  Seed for subsampling and HVG.

    Returns:
        AnnData with log-normalised counts, adata.var["highly_variable"] mask,
        adata.obs["ct_label"] integer labels, and adata.uns["celltypes"] list.
    """
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    path = root / SCRNA_FILES[region] if region in SCRNA_FILES else Path(region)
    _check_path(path)

    print(f"[scRNA-seq] Loading {path.name} ...")
    adata = sc.read_h5ad(path)
    adata = _ensure_raw_counts(adata)
    adata.X = adata.X.astype(np.float32)

    if max_cells and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=random_state)
        print(f"  Subsampled → {max_cells:,} cells")

    print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _remove_mito(adata)

    # HVG selection — necessary for 32k gene input
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", span=1.0)
    print(f"  HVG selected: {adata.var['highly_variable'].sum()}")

    _attach_celltypes(adata, celltype_key, root / CELL_METADATA_FILE)
    print(f"  Cell types ({celltype_key}): {len(adata.uns['celltypes'])}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# MERFISH loader  (NO HVG — all ~500 genes used directly)
# ─────────────────────────────────────────────────────────────────────────────

def load_abc_merfish(
    name: str = "MERFISH-C57BL6J-638850",
    data_root: str | Path | None = None,
    celltype_key: str = "subclass",
    brain_section: str | None = None,
    max_cells: int | None = None,
    random_state: int = 0,
    backed: bool = True,
) -> sc.AnnData:
    """Load the MERFISH whole mouse brain dataset.

    HVG selection is NOT applied. The ABC tutorial matrix contains 500
    measured genes plus 50 blank codewords; blank codewords are excluded here
    so the compression universe is the 500 real MERFISH genes.

    Spatial coordinates (x, y, z) are stored in adata.obsm["spatial"].

    Args:
        name:           MERFISH dataset key or a direct path to a log-normalised h5ad.
        data_root:      Root directory.
        celltype_key:   Column in obs for cell type labels.
        brain_section:  Optional: filter to a single coronal section by z-index
                        (string like "C57BL6J-638850.46").  None = whole brain.
        max_cells:      Optional cell subsampling limit.
        random_state:   Seed for subsampling.
        backed:         Open the full h5ad in backed mode and materialize only
                        the selected cells and real genes.

    Returns:
        AnnData with:
          - adata.X              log2-normalised counts, shape (cells, 500)
          - adata.var_names      the 500 MERFISH gene identifiers
          - adata.obsm["spatial"]  (cells, 3) spatial coordinates
          - adata.obs["ct_label"]  integer cell type labels
          - adata.uns["celltypes"] ordered list of cell type names
          - adata.uns["is_merfish"] = True  (flag for downstream code)
    """
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    path = _resolve_merfish_expression_path(root, name)
    _check_path(path)

    print(f"[MERFISH] Loading {path.name} ...")
    adata = sc.read_h5ad(path, backed="r" if backed else None)
    obs = adata.obs.copy()
    var = adata.var.copy()

    print(f"  Raw shape: {adata.n_obs:,} cells x {adata.n_vars} variables")

    metadata = _load_merfish_cell_metadata(root)
    if metadata is not None:
        obs = _join_obs_metadata(obs, metadata)
        print(f"  Joined MERFISH metadata: {obs.shape[1]} obs columns")

    # Optional: filter to a single brain section before materializing X.
    if brain_section is not None:
        sec_key = _find_section_key_from_obs(obs)
        obs = obs.loc[obs[sec_key] == brain_section]
        print(f"  Filtered to section '{brain_section}': {len(obs):,} cells")

    if max_cells and len(obs) > max_cells:
        obs = obs.sample(n=max_cells, random_state=random_state).sort_index()
        print(f"  Subsampled → {max_cells:,} cells")

    real_gene_mask = _real_merfish_gene_mask(var)
    n_blanks = int((~real_gene_mask).sum())
    adata.uns["n_blank_codewords_removed"] = n_blanks
    if n_blanks:
        print(f"  Excluding {n_blanks} blank codewords")

    # The ABC tutorial uses the C57BL6J-638850/log2 file. Do not re-normalize.
    adata = adata[obs.index, var.index[real_gene_mask]].to_memory()
    adata.obs = obs.loc[adata.obs_names].copy()
    adata.var = var.loc[adata.var_names].copy()
    adata.X = adata.X.astype(np.float32)

    # Mark all genes as "selected" (no HVG concept here)
    adata.var["highly_variable"] = True   # so downstream code still works

    _attach_spatial_coords(adata)
    _attach_celltypes(adata, celltype_key, root / MERFISH_METADATA_FILE)
    adata.uns["is_merfish"] = True
    adata.uns["n_spatial_genes"] = adata.n_vars
    adata.uns["expression_file"] = str(path)

    print(f"  Final shape: {adata.n_obs:,} cells x {adata.n_vars} real MERFISH genes")
    print(f"  Cell types ({celltype_key}): {len(adata.uns['celltypes'])}")
    print(f"  Spatial coords: {adata.obsm['spatial'].shape}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point — auto-detects scRNA-seq vs. MERFISH
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    name: str,
    data_root: str | Path | None = None,
    celltype_key: str = "subclass",
    n_hvg: int = 3000,
    max_cells: int | None = None,
    random_state: int = 0,
    **kwargs,
) -> sc.AnnData:
    """Unified loader: dispatches to scRNA-seq or MERFISH loader automatically.

    Args:
        name:  One of the SCRNA_FILES keys, a MERFISH_FILES key, or a file path.
               MERFISH datasets are auto-detected by name prefix "MERFISH-".

    Returns:
        Preprocessed AnnData ready for GeneCompressionAE.
    """
    if name.startswith("MERFISH") or name in MERFISH_FILES:
        return load_abc_merfish(
            name=name,
            data_root=data_root,
            celltype_key=celltype_key,
            max_cells=max_cells,
            random_state=random_state,
            **kwargs,
        )
    else:
        return load_abc_scrna(
            region=name,
            data_root=data_root,
            celltype_key=celltype_key,
            n_hvg=n_hvg,
            max_cells=max_cells,
            random_state=random_state,
        )


def dataset_info(adata: sc.AnnData) -> dict:
    """Return a summary dict describing the loaded dataset."""
    is_merfish = adata.uns.get("is_merfish", False)
    n_genes = adata.n_vars if is_merfish else int(adata.var["highly_variable"].sum())
    return {
        "modality": "MERFISH" if is_merfish else "scRNA-seq",
        "n_cells": adata.n_obs,
        "n_genes_input": n_genes,
        "has_spatial": "spatial" in adata.obsm,
        "n_celltypes": len(adata.uns.get("celltypes", [])),
        "hvg_applied": not is_merfish,
        "n_blank_codewords_removed": int(adata.uns.get("n_blank_codewords_removed", 0)),
        "note": (
            "All real MERFISH genes used (targeted panel, no HVG; blanks removed)"
            if is_merfish
            else f"HVG-filtered to {n_genes} from full transcriptome"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset and DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class AnnDataset(Dataset):
    """PyTorch Dataset wrapping an AnnData object.

    Works for both scRNA-seq (with HVG mask applied) and MERFISH (all genes).
    For MERFISH, spatial coordinates are also returned if available.

    Args:
        adata:           Preprocessed AnnData.
        genes_key:       adata.var column for gene mask.  For MERFISH this is
                         "highly_variable" (all True), for scRNA-seq it filters
                         to HVGs.
        ct_key:          adata.obs column for integer cell type labels.
        return_spatial:  Also return spatial (x, y, z) coordinates.
    """

    def __init__(
        self,
        adata: sc.AnnData,
        genes_key: str = "highly_variable",
        ct_key: str = "ct_label",
        return_spatial: bool = False,
    ):
        import scipy.sparse as sp

        # Apply gene mask
        if genes_key in adata.var.columns:
            a = adata[:, adata.var[genes_key]]
        else:
            a = adata

        X = a.X.toarray() if sp.issparse(a.X) else np.array(a.X)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.ct = torch.from_numpy(adata.obs[ct_key].values.astype(np.int64))
        self.var_names = list(a.var_names)
        self.obs_names = list(adata.obs_names)

        self.spatial = None
        if return_spatial and "spatial" in adata.obsm:
            coords = adata.obsm["spatial"].astype(np.float32)
            self.spatial = torch.from_numpy(coords)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        item = {"x": self.X[idx], "ct": self.ct[idx]}
        if self.spatial is not None:
            item["spatial"] = self.spatial[idx]
        return item


def make_loaders(
    adata: sc.AnnData,
    genes_key: str = "highly_variable",
    ct_key: str = "ct_label",
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    batch_size: int = 512,
    num_workers: int = 4,
    random_state: int = 0,
    return_spatial: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split adata into train / val / test DataLoaders.

    For MERFISH data, spatial coordinates are included in each batch if
    return_spatial=True (or auto-enabled when adata.uns["is_merfish"] is set).

    Returns:
        train_loader, val_loader, test_loader
    """
    if adata.uns.get("is_merfish", False):
        return_spatial = True   # always return spatial for MERFISH

    dataset = AnnDataset(adata, genes_key=genes_key, ct_key=ct_key,
                         return_spatial=return_spatial)
    n = len(dataset)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(random_state)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    is_merfish = adata.uns.get("is_merfish", False)
    print(
        f"DataLoaders ready — modality: {'MERFISH (spatial)' if is_merfish else 'scRNA-seq'}\n"
        f"  train: {n_train:,}  val: {n_val:,}  test: {n_test:,}  "
        f"batch_size: {batch_size}"
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Set the ABC_ATLAS_PATH environment variable or pass data_root."
        )


def _resolve_merfish_expression_path(root: Path, name: str) -> Path:
    """Resolve ABC tutorial-style MERFISH log2 h5ad paths."""
    direct = Path(name)
    if direct.suffix == ".h5ad" or direct.exists():
        return direct
    if name in MERFISH_FILES:
        candidate = root / MERFISH_FILES[name]
        if candidate.exists():
            return candidate

    candidates = [
        root / "expression_matrices" / name / "20230830" / "C57BL6J-638850-log2.h5ad",
        root / "expression_matrices" / name / "20231215" / "C57BL6J-638850-log2.h5ad",
        root / "spatial" / name / "20231215" / "C57BL6J-638850.h5ad",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted((root / "expression_matrices").glob(f"{name}/**/*log2*.h5ad"))
    if matches:
        return matches[0]
    return root / MERFISH_FILES.get(name, name)


def _load_merfish_cell_metadata(root: Path) -> pd.DataFrame | None:
    """Load rich MERFISH cell metadata when available.

    The ABC notebook constructs this by joining cell metadata with taxonomy
    tables, and the cache also exposes it as cell_metadata_with_cluster_annotation.
    """
    candidates = [
        root / MERFISH_METADATA_FILE,
        root / "metadata" / "MERFISH-C57BL6J-638850" / "20231215" / "cell_metadata_with_cluster_annotation.csv",
    ]
    candidates += sorted(root.glob("metadata/MERFISH-C57BL6J-638850/**/cell_metadata_with_cluster_annotation*.csv"))
    candidates += sorted(root.glob("**/MERFISH-C57BL6J-638850/**/cell_metadata_with_cluster_annotation*.csv"))

    for path in candidates:
        if path.exists():
            meta = pd.read_csv(path, index_col=0, low_memory=False)
            meta.index = meta.index.astype(str)
            return meta

    cell = _read_first_existing_csv(
        [
            root / "metadata" / "MERFISH-C57BL6J-638850" / "20230830" / "cell_metadata.csv",
            root / "metadata" / "MERFISH-C57BL6J-638850" / "20231215" / "cell_metadata.csv",
        ]
        + sorted(root.glob("metadata/MERFISH-C57BL6J-638850/**/cell_metadata.csv"))
    )
    if cell is None:
        return None

    cluster_details = _read_first_existing_csv(
        [
            root / "metadata" / "WMB-taxonomy" / "20231215" / "cluster_to_cluster_annotation_membership_pivoted.csv",
        ]
        + sorted(root.glob("metadata/WMB-taxonomy/**/cluster_to_cluster_annotation_membership_pivoted.csv"))
    )
    cluster_colors = _read_first_existing_csv(
        [
            root / "metadata" / "WMB-taxonomy" / "20231215" / "cluster_to_cluster_annotation_membership_color.csv",
        ]
        + sorted(root.glob("metadata/WMB-taxonomy/**/cluster_to_cluster_annotation_membership_color.csv"))
    )

    if cluster_details is not None and "cluster_alias" in cell.columns:
        if "cluster_alias" in cluster_details.columns:
            cluster_details = cluster_details.set_index("cluster_alias", drop=False)
        cell = cell.join(cluster_details, on="cluster_alias", rsuffix="_taxonomy")
    if cluster_colors is not None and "cluster_alias" in cell.columns:
        if "cluster_alias" in cluster_colors.columns:
            cluster_colors = cluster_colors.set_index("cluster_alias", drop=False)
        cell = cell.join(cluster_colors, on="cluster_alias", rsuffix="_color")

    cell.index = cell.index.astype(str)
    return cell


def _read_first_existing_csv(paths: list[Path]) -> pd.DataFrame | None:
    for path in paths:
        if path.exists():
            df = pd.read_csv(path, index_col=0, low_memory=False)
            df.index = df.index.astype(str)
            return df
    return None


def _join_obs_metadata(obs: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join metadata by cell label without clobbering existing h5ad obs columns."""
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    shared = obs.index.intersection(metadata.index)
    if shared.empty:
        raise ValueError("No shared cell labels between MERFISH h5ad obs and metadata table.")

    obs = obs.loc[shared]
    meta = metadata.loc[shared]
    new_cols = [col for col in meta.columns if col not in obs.columns]
    if new_cols:
        obs = obs.join(meta[new_cols])
    for col in meta.columns.intersection(obs.columns):
        obs[col] = obs[col].combine_first(meta[col])
    return obs


def _real_merfish_gene_mask(var: pd.DataFrame) -> pd.Series:
    """Return True for measured genes and False for blank MERFISH codewords."""
    symbols = var.get("gene_symbol", pd.Series(index=var.index, data="")).fillna("").astype(str)
    ids = pd.Series(var.index, index=var.index).fillna("").astype(str)
    names = var.get("gene_name", pd.Series(index=var.index, data="")).fillna("").astype(str)

    blank = (
        symbols.str.contains(r"blank|codeword", case=False, regex=True)
        | ids.str.contains(r"blank|codeword", case=False, regex=True)
        | names.str.contains(r"blank|codeword", case=False, regex=True)
        | symbols.eq("")
    )
    mask = ~blank
    if mask.sum() == len(mask):
        print("  Warning: no blank codewords detected from var metadata")
    return mask


def _ensure_raw_counts(adata: sc.AnnData) -> sc.AnnData:
    """Move adata.raw → adata.X if it looks more like raw counts."""
    if adata.raw is not None:
        raw_max = adata.raw.X.data.max() if hasattr(adata.raw.X, "data") else adata.raw.X.max()
        x_max = adata.X.data.max() if hasattr(adata.X, "data") else adata.X.max()
        if raw_max > x_max:
            adata = adata.raw.to_adata()
    return adata


def _looks_like_raw_counts(adata: sc.AnnData) -> bool:
    """Heuristic: raw integer counts have max < 500 and many exact integers."""
    import scipy.sparse as sp
    X = adata.X
    sample = X[:1000].toarray() if sp.issparse(X) else X[:1000]
    return float(sample.max()) < 500 and float((sample % 1 == 0).mean()) > 0.9


def _remove_mito(adata: sc.AnnData) -> None:
    """Remove mitochondrial genes in-place."""
    mito = adata.var_names.str.startswith(("mt-", "MT-"))
    if mito.any():
        adata._inplace_subset_var(~mito)


def _attach_spatial_coords(adata: sc.AnnData) -> None:
    """Ensure spatial coordinates are in adata.obsm['spatial']."""
    if "spatial" not in adata.obsm:
        coord_cols = [c for c in ["x", "y", "z"] if c in adata.obs.columns]
        if coord_cols:
            adata.obsm["spatial"] = adata.obs[coord_cols].values.astype(np.float32)


def _find_section_key(adata: sc.AnnData) -> str:
    for col in ["brain_section_label", "section", "slice"]:
        if col in adata.obs.columns:
            return col
    raise KeyError("Could not find a brain section column in adata.obs")


def _find_section_key_from_obs(obs: pd.DataFrame) -> str:
    for col in ["brain_section_label", "section", "slice"]:
        if col in obs.columns:
            return col
    raise KeyError("Could not find a brain section column in obs metadata")


def _attach_celltypes(adata: sc.AnnData, celltype_key: str, meta_path: Path) -> None:
    """Add integer ct_label column; load from metadata CSV if key missing."""
    if celltype_key not in adata.obs.columns:
        if meta_path.exists():
            meta = pd.read_csv(meta_path, index_col=0, low_memory=False)
            shared = adata.obs.index.intersection(meta.index)
            adata._inplace_subset_obs(shared)
            adata.obs[celltype_key] = meta.loc[shared, celltype_key]
        else:
            raise KeyError(
                f"Cell type key '{celltype_key}' not in adata.obs and "
                f"metadata file not found at {meta_path}"
            )

    # Drop NaN cell types
    valid = adata.obs[celltype_key].notna()
    if not valid.all():
        adata._inplace_subset_obs(valid)

    celltypes = sorted(adata.obs[celltype_key].unique())
    ct_to_int = {ct: i for i, ct in enumerate(celltypes)}
    adata.obs["ct_label"] = adata.obs[celltype_key].map(ct_to_int).astype(int)
    adata.uns["celltypes"] = celltypes
    adata.uns["ct_to_int"] = ct_to_int
    adata.uns["celltype_key"] = celltype_key
