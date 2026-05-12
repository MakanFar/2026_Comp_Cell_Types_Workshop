#!/usr/bin/env python3
"""
Full comparison pipeline: train all panel sizes × all methods, evaluate, visualize.

This is the master script for the compression experiment.  Run it after
downloading / mounting the ABC Atlas data.

Usage (Code Ocean):
    python scripts/compare_panels.py

Usage (local):
    ABC_ATLAS_PATH=/path/to/data python scripts/compare_panels.py \\
        --region WMB-10Xv3-TH \\
        --panel_sizes 10 25 50 100 \\
        --max_cells 50000

Output layout (all written to save_dir / results_dir):
    results/
    ├── {region}/
    │   ├── baselines/
    │   │   ├── baseline_genes.json          per-method per-size gene lists
    │   │   └── baseline_eval.csv            evaluation metrics
    │   ├── trained_models/
    │   │   ├── concrete_k10/                model checkpoint + selected genes
    │   │   ├── concrete_k25/
    │   │   ├── concrete_k50/
    │   │   └── concrete_k100/
    │   ├── eval_summary.csv                 all methods × all sizes × all metrics
    │   └── figures/
    │       ├── umap_comparison_k50.png
    │       ├── metric_comparison.png
    │       └── reconstruction_scatter_k50.png
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.abc_atlas import load_dataset, AnnDataset
from src.baselines.gene_selection import run_all_baselines
from src.evaluation.metrics import compare_panels, evaluate_panel
from src.visualization.plots import plot_panel_comparison, plot_umap_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full gene compression comparison")
    p.add_argument("--data_root", default=os.environ.get("ABC_ATLAS_PATH", "/data/abc_atlas"))
    p.add_argument("--region", default="WMB-10Xv3-TH")
    p.add_argument("--celltype_key", default="subclass")
    p.add_argument("--n_hvg", type=int, default=3000)
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--brain_section", default=None,
                   help="MERFISH only: restrict to one section, e.g. C57BL6J-638850.38")
    p.add_argument("--panel_sizes", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--save_dir", default=os.environ.get("RESULTS_DIR", "/results"))
    p.add_argument("--skip_train", action="store_true",
                   help="Skip encoder training; only run baseline comparisons")
    p.add_argument("--skip_spapros", action="store_true",
                   help="Skip Spapros selection (slow; requires spapros installed)")
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load trained model and get selected genes
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_genes(model_dir: Path) -> list[str]:
    genes_path = model_dir / "selected_genes.json"
    if genes_path.exists():
        with open(genes_path) as f:
            return json.load(f)
    return []


def get_latent_embeddings(
    model_dir: Path,
    adata,
    device: str = "cpu",
) -> np.ndarray | None:
    """Load checkpoint and extract latent embeddings for all cells."""
    try:
        sys.path.insert(0, str(model_dir.parent.parent))
        from src.models.autoencoder import build_model

        with open(model_dir / "args.json") as f:
            cfg_args = json.load(f)
        with open(model_dir / "celltypes.json") as f:
            celltypes = json.load(f)

        cfg = {
            "n_genes": int(adata.var["highly_variable"].sum()),
            "n_selected": cfg_args["n_selected"],
            "n_celltypes": len(celltypes),
            "latent_dim": cfg_args["latent_dim"],
            "encoder_dims": cfg_args["encoder_dims"],
            "decoder_dims": cfg_args["decoder_dims"],
            "selector_type": cfg_args["selector_type"],
        }
        model = build_model(cfg).to(device)
        ckpt = torch.load(model_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        dataset = AnnDataset(adata)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        all_h = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                out = model(x)
                all_h.append(out["h"].cpu().numpy())
        return np.vstack(all_h)

    except Exception as e:
        print(f"  Could not extract embeddings: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.save_dir) / args.region
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir = out_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" Gene Panel Compression: {args.region}")
    print(f" Panel sizes: {args.panel_sizes}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data ...")
    adata = load_dataset(
        name=args.region,
        data_root=args.data_root,
        celltype_key=args.celltype_key,
        n_hvg=args.n_hvg,
        max_cells=args.max_cells,
        random_state=args.seed,
        brain_section=args.brain_section,
    )

    # ── 2. Baseline gene selections ───────────────────────────────────────────
    print("\n[2/5] Running baseline gene selections ...")
    baselines = run_all_baselines(
        adata,
        panel_sizes=args.panel_sizes,
        celltype_key=args.celltype_key,
        include_spapros=not args.skip_spapros,
        spapros_save_dir=str(baselines_dir / "spapros"),
        n_jobs=args.n_jobs,
    )

    # Save baseline gene lists
    with open(baselines_dir / "baseline_genes.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # ── 3. Train encoder models (one per panel size) ──────────────────────────
    if not args.skip_train:
        print("\n[3/5] Training GeneCompressionAE models ...")
        for n in args.panel_sizes:
            print(f"\n  Training k={n} ...")
            cmd = [
                sys.executable, str(Path(__file__).parent / "train.py"),
                "--region", args.region,
                "--data_root", args.data_root,
                "--celltype_key", args.celltype_key,
                "--n_hvg", str(args.n_hvg),
                "--n_selected", str(n),
                "--n_epochs", str(args.n_epochs),
                "--batch_size", str(args.batch_size),
                "--save_dir", str(models_dir),
                "--seed", str(args.seed),
            ]
            if args.max_cells:
                cmd += ["--max_cells", str(args.max_cells)]
            if args.brain_section:
                cmd += ["--brain_section", args.brain_section]
            subprocess.run(cmd, check=True)
    else:
        print("\n[3/5] Skipping encoder training (--skip_train)")

    # ── 4. Evaluate all methods ───────────────────────────────────────────────
    print("\n[4/5] Evaluating all gene panels ...")

    all_results = []

    for n in args.panel_sizes:
        print(f"\n  Panel size: {n}")

        # Collect all gene panels for this size
        panels: dict[str, list[str]] = {}

        # Baselines
        for method, size_dict in baselines.items():
            if n in size_dict:
                panels[method] = size_dict[n]

        # Trained model (concrete AE)
        model_dir = models_dir / f"{args.region}__concrete_k{n}"
        ae_genes = load_trained_genes(model_dir)
        if ae_genes:
            panels["concrete_ae"] = ae_genes

        # Evaluate
        for method, genes in panels.items():
            print(f"    Evaluating {method} ({len(genes)} genes) ...")

            embeddings = None
            if method == "concrete_ae" and model_dir.exists():
                embeddings = get_latent_embeddings(model_dir, adata)

            try:
                metrics = evaluate_panel(
                    adata, genes,
                    embeddings=embeddings,
                    celltype_key="ct_label",
                    run_clustering=(n >= 25),   # clustering is slow for k=10
                    run_knn=True,
                    run_ct_f1=True,
                )
                metrics["method"] = method
                metrics["n_genes"] = n
                all_results.append(metrics)
            except Exception as e:
                print(f"    ERROR evaluating {method}: {e}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_root / "eval_summary.csv", index=False)
    print(f"\nResults saved to {out_root / 'eval_summary.csv'}")

    # ── 5. Visualize ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating figures ...")

    # Main comparison bar chart
    if len(results_df) > 0:
        key_metrics = ["f1_macro", "knn_overlap_mean", "nmi_mean"]
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        fig = plot_panel_comparison(
            results_df,
            metrics=available_metrics,
            save_path=str(figures_dir / "metric_comparison.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved: {figures_dir / 'metric_comparison.png'}")

    # UMAP comparison for k=50 (or largest available size)
    focus_k = max([k for k in args.panel_sizes if k <= 50], default=args.panel_sizes[0])
    focus_method = "concrete_ae" if not args.skip_train else "spapros"
    if focus_method in baselines or not args.skip_train:
        genes_50 = (
            load_trained_genes(models_dir / f"{args.region}__concrete_k{focus_k}")
            if not args.skip_train
            else baselines.get("spapros", {}).get(focus_k, [])
        )
        if genes_50:
            import scanpy as sc
            genes_in = [g for g in genes_50 if g in adata.var_names]
            adata_panel = adata[:, genes_in].copy()
            adata_full = adata[:, adata.var["highly_variable"]].copy()

            embeddings_50 = None
            if not args.skip_train:
                embeddings_50 = get_latent_embeddings(
                    models_dir / f"{args.region}__concrete_k{focus_k}", adata
                )

            fig = plot_umap_comparison(
                adata_full, adata_panel, embeddings_50,
                celltype_key=args.celltype_key,
                save_path=str(figures_dir / f"umap_comparison_k{focus_k}.png"),
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  Saved: {figures_dir / f'umap_comparison_k{focus_k}.png'}")

    print(f"\n{'=' * 60}")
    print(f" Complete!  All outputs in: {out_root}")
    print(f"{'=' * 60}")

    # Print summary table
    if len(results_df) > 0:
        summary_cols = ["method", "n_genes"] + [
            c for c in ["f1_macro", "knn_overlap_mean", "nmi_mean"]
            if c in results_df.columns
        ]
        print("\nSummary:")
        print(results_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
