"""Notebook-friendly workflows for the gene compression experiment.

These helpers train learned panels, run baselines, evaluate results, and save
outputs from an AnnData object that you already loaded in a notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines.gene_selection import run_all_baselines
from .data.abc_atlas import make_loaders
from .evaluation.metrics import evaluate_panel
from .models.autoencoder import build_model
from .training.losses import CompressionLoss
from .training.trainer import TemperatureScheduler, Trainer


def get_device(device: str | None = None) -> str:
    """Return the requested or best available torch device."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def input_genes(adata) -> list[str]:
    """Return the model input gene identifiers."""
    if "highly_variable" in adata.var.columns:
        return adata.var_names[adata.var["highly_variable"]].tolist()
    return adata.var_names.tolist()


def gene_symbol_map(adata) -> dict[str, str]:
    """Map gene identifiers to symbols when the metadata is available."""
    if "gene_symbol" in adata.var.columns:
        return adata.var["gene_symbol"].astype(str).to_dict()
    return {gene: gene for gene in adata.var_names}


def train_panel(
    adata,
    n_selected: int,
    output_dir: str | Path,
    *,
    selector_type: str = "concrete",
    latent_dim: int = 64,
    encoder_dims: tuple[int, ...] = (256, 128),
    decoder_dims: tuple[int, ...] = (128, 256),
    dropout: float = 0.1,
    n_epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    early_stop: int = 15,
    num_workers: int = 0,
    seed: int = 0,
    device: str | None = None,
    lambda_recon: float = 1.0,
    lambda_ct: float = 0.5,
    lambda_neighbor: float = 0.0,
    lambda_stg: float = 0.01,
    t_start: float = 1.0,
    t_min: float = 0.01,
    anneal_rate: float = 3e-4,
    anneal_every: int = 100,
) -> dict[str, Any]:
    """Train one learned panel and return notebook-friendly results.

    Returns a dict containing the trained model, history, metrics, selected gene
    identifiers, selected gene symbols, and output directory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    genes = input_genes(adata)
    if n_selected > len(genes):
        raise ValueError(f"n_selected={n_selected} exceeds input gene count {len(genes)}")

    run_dir = Path(output_dir) / f"{selector_type}_k{n_selected}"
    run_dir.mkdir(parents=True, exist_ok=True)

    symbols = gene_symbol_map(adata)
    with open(run_dir / "input_genes.json", "w") as f:
        json.dump(genes, f, indent=2)
    with open(run_dir / "gene_symbols.json", "w") as f:
        json.dump(symbols, f, indent=2)
    with open(run_dir / "celltypes.json", "w") as f:
        json.dump(adata.uns["celltypes"], f, indent=2)

    train_loader, val_loader, test_loader = make_loaders(
        adata,
        batch_size=batch_size,
        num_workers=num_workers,
        random_state=seed,
    )

    cfg = {
        "n_genes": len(genes),
        "n_selected": n_selected,
        "n_celltypes": len(adata.uns["celltypes"]),
        "latent_dim": latent_dim,
        "encoder_dims": list(encoder_dims),
        "decoder_dims": list(decoder_dims),
        "selector_type": selector_type,
        "temperature": t_start,
        "dropout": dropout,
    }
    model = build_model(cfg)
    device = get_device(device)

    loss_fn = CompressionLoss(
        lambda_recon=lambda_recon,
        lambda_ct=lambda_ct,
        lambda_neighbor=lambda_neighbor,
        lambda_stg=lambda_stg if selector_type == "stg" else 0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    temp_scheduler = TemperatureScheduler(
        t_start=t_start,
        t_min=t_min,
        anneal_rate=anneal_rate,
        anneal_every=anneal_every,
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        temp_scheduler=temp_scheduler,
        device=device,
        save_dir=str(run_dir),
        grad_clip=grad_clip,
    )

    history = trainer.fit(
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        early_stop_patience=early_stop,
    )
    test_metrics = trainer.evaluate(test_loader)

    selected_genes = model.get_selected_genes(var_names=genes)
    selected_symbols = [symbols.get(gene, gene) for gene in selected_genes]

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "args.json", "w") as f:
        json.dump({**cfg, "seed": seed, "batch_size": batch_size}, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(run_dir / "selected_genes.json", "w") as f:
        json.dump(selected_genes, f, indent=2)
    with open(run_dir / "selected_gene_symbols.json", "w") as f:
        json.dump(selected_symbols, f, indent=2)

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "selected_genes": selected_genes,
        "selected_gene_symbols": selected_symbols,
        "run_dir": run_dir,
    }


def train_panel_sizes(
    adata,
    panel_sizes: list[int] = [10, 25, 50, 100],
    output_dir: str | Path = "results/trained_models",
    **kwargs,
) -> dict[int, dict[str, Any]]:
    """Train learned panels for multiple panel sizes."""
    runs: dict[int, dict[str, Any]] = {}
    for n_selected in panel_sizes:
        print(f"\nTraining learned panel k={n_selected}")
        runs[n_selected] = train_panel(adata, n_selected, output_dir, **kwargs)
    return runs


def learned_panels(runs: dict[int, dict[str, Any]], method_name: str = "concrete_ae") -> dict[str, dict[int, list[str]]]:
    """Convert train_panel_sizes output to the baseline panel dict format."""
    return {method_name: {n: run["selected_genes"] for n, run in runs.items()}}


def run_baselines(
    adata,
    panel_sizes: list[int] = [10, 25, 50, 100],
    *,
    celltype_key: str = "subclass",
    include_spapros: bool = False,
    output_dir: str | Path | None = None,
    random_seeds: list[int] | None = None,
    n_jobs: int = -1,
) -> dict[str, dict[int, list[str]]]:
    """Run PCA, DE, HVG/PCA fallback, random, and optionally Spapros baselines."""
    return run_all_baselines(
        adata,
        panel_sizes=panel_sizes,
        celltype_key=celltype_key,
        include_spapros=include_spapros,
        spapros_save_dir=str(output_dir) if output_dir else None,
        random_seeds=random_seeds,
        n_jobs=n_jobs,
    )


def merge_panel_dicts(*panel_dicts: dict[str, dict[int, list[str]]]) -> dict[str, dict[int, list[str]]]:
    """Merge nested panel dictionaries of the form {method: {k: genes}}."""
    merged: dict[str, dict[int, list[str]]] = {}
    for panel_dict in panel_dicts:
        for method, size_dict in panel_dict.items():
            merged.setdefault(method, {}).update(size_dict)
    return merged


def evaluate_panels(
    adata,
    panels: dict[str, dict[int, list[str]]],
    *,
    celltype_key: str = "ct_label",
    run_clustering_min_k: int | None = 25,
    run_knn: bool = True,
    run_ct_f1: bool = True,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate all methods and panel sizes into a tidy DataFrame."""
    rows = []
    for method, size_dict in panels.items():
        for n_genes, genes in sorted(size_dict.items()):
            print(f"Evaluating {method} k={n_genes}")
            metrics = evaluate_panel(
                adata,
                genes,
                celltype_key=celltype_key,
                run_clustering=run_clustering_min_k is None or n_genes >= run_clustering_min_k,
                run_knn=run_knn,
                run_ct_f1=run_ct_f1,
            )
            metrics["method"] = method
            metrics["n_genes"] = n_genes
            rows.append(metrics)

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df
