#!/usr/bin/env python3
"""
Train a GeneCompressionAE for a single panel size.

Usage (Code Ocean):
    python scripts/train.py --region WMB-10Xv3-TH --n_selected 50

Usage (local):
    ABC_ATLAS_PATH=/path/to/data python scripts/train.py --n_selected 50

Checkpoints and logs go to /results/ (Code Ocean) or --save_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# ── Make src importable regardless of working directory ───────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.abc_atlas import load_dataset, make_loaders, dataset_info
from src.models.autoencoder import build_model
from src.training.losses import CompressionLoss
from src.training.trainer import Trainer, TemperatureScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GeneCompressionAE")

    # Data
    p.add_argument("--data_root", default=os.environ.get("ABC_ATLAS_PATH", "/data/abc_atlas"))
    p.add_argument("--region", default="WMB-10Xv3-TH",
                   help="ABC Atlas region key or direct path to h5ad file")
    p.add_argument("--celltype_key", default="subclass",
                   choices=["class", "subclass", "supertype", "cluster"])
    p.add_argument("--n_hvg", type=int, default=3000)
    p.add_argument("--max_cells", type=int, default=None,
                   help="Subsample to this many cells (None = use all)")
    p.add_argument("--brain_section", default=None,
                   help="MERFISH only: restrict to one section, e.g. C57BL6J-638850.38")

    # Model
    p.add_argument("--n_selected", type=int, default=50,
                   help="Gene panel size k (10 / 25 / 50 / 100)")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--encoder_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--decoder_dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--selector_type", default="concrete", choices=["concrete", "stg"])
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--n_epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=15)

    # Loss weights
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_ct", type=float, default=0.5)
    p.add_argument("--lambda_neighbor", type=float, default=0.0,
                   help="Triplet neighbor loss weight (0 = disabled; expensive)")
    p.add_argument("--lambda_stg", type=float, default=0.01,
                   help="STG L0 regularisation weight (ignored for concrete)")

    # Temperature annealing (Concrete only)
    p.add_argument("--t_start", type=float, default=1.0)
    p.add_argument("--t_min", type=float, default=0.01)
    p.add_argument("--anneal_rate", type=float, default=3e-4)
    p.add_argument("--anneal_every", type=int, default=100)

    # I/O
    p.add_argument("--save_dir", default=os.environ.get("RESULTS_DIR", "/results"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    run_name = f"{args.region}__{args.selector_type}_k{args.n_selected}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # ── Data ─────────────────────────────────────────────────────────────────
    adata = load_dataset(
        name=args.region,
        data_root=args.data_root,
        celltype_key=args.celltype_key,
        n_hvg=args.n_hvg,
        max_cells=args.max_cells,
        random_state=args.seed,
        brain_section=args.brain_section,
    )

    info = dataset_info(adata)
    print(json.dumps(info, indent=2))

    # For MERFISH: n_genes = all ~500 genes (no HVG mask applied)
    # For scRNA-seq: n_genes = HVG subset
    n_genes = info["n_genes_input"]
    n_celltypes = len(adata.uns["celltypes"])
    print(f"n_genes={n_genes}  n_celltypes={n_celltypes}")

    # Save input gene identifiers for later gene retrieval. For MERFISH this is
    # the 500 real measured genes after blank codeword removal.
    hvg_names = adata.var_names[adata.var["highly_variable"]].tolist()
    gene_symbols = (
        adata.var["gene_symbol"].astype(str).to_dict()
        if "gene_symbol" in adata.var.columns
        else {gene: gene for gene in hvg_names}
    )
    with open(save_dir / "hvg_names.json", "w") as f:
        json.dump(hvg_names, f)
    with open(save_dir / "gene_symbols.json", "w") as f:
        json.dump(gene_symbols, f, indent=2)
    with open(save_dir / "celltypes.json", "w") as f:
        json.dump(adata.uns["celltypes"], f)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, test_loader = make_loaders(
        adata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = {
        "n_genes": n_genes,
        "n_selected": args.n_selected,
        "n_celltypes": n_celltypes,
        "latent_dim": args.latent_dim,
        "encoder_dims": args.encoder_dims,
        "decoder_dims": args.decoder_dims,
        "selector_type": args.selector_type,
        "temperature": args.t_start,
        "dropout": args.dropout,
    }
    model = build_model(cfg)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Loss, optimizer, schedulers ──────────────────────────────────────────
    loss_fn = CompressionLoss(
        lambda_recon=args.lambda_recon,
        lambda_ct=args.lambda_ct,
        lambda_neighbor=args.lambda_neighbor,
        lambda_stg=args.lambda_stg if args.selector_type == "stg" else 0.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs
    )
    temp_scheduler = TemperatureScheduler(
        t_start=args.t_start,
        t_min=args.t_min,
        anneal_rate=args.anneal_rate,
        anneal_every=args.anneal_every,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        temp_scheduler=temp_scheduler,
        device=device,
        save_dir=str(save_dir),
        grad_clip=args.grad_clip,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n═══ Starting training ═══")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=args.n_epochs,
        early_stop_patience=args.early_stop,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n═══ Test set evaluation ═══")
    test_metrics = trainer.evaluate(test_loader)
    print(json.dumps(test_metrics, indent=2))
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Extract selected genes ────────────────────────────────────────────────
    selected_genes = model.get_selected_genes(var_names=hvg_names)
    selected_gene_symbols = [gene_symbols.get(gene, gene) for gene in selected_genes]
    print(f"\nSelected {len(selected_genes)} genes:")
    print(selected_genes)
    with open(save_dir / "selected_genes.json", "w") as f:
        json.dump(selected_genes, f, indent=2)
    with open(save_dir / "selected_gene_symbols.json", "w") as f:
        json.dump(selected_gene_symbols, f, indent=2)

    # ── Save training curves ──────────────────────────────────────────────────
    import pandas as pd
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)

    try:
        from src.visualization.plots import plot_training_history
        fig = plot_training_history(history, save_path=str(save_dir / "training_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Could not save training plot: {e}")

    print(f"\nDone. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
