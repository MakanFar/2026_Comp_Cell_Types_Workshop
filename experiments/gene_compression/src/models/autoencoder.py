"""
Full Concrete Autoencoder for gene panel compression.

Architecture
------------
                         ┌─────────────┐
  x (n_genes) ──────────▶│ ConcreteSelector │──▶ z_sel (k)
                         └─────────────┘
                                │
                         ┌─────────────┐
                         │   Encoder   │──▶ h (latent_dim)
                         └─────────────┘
                           ┌───┴───┐
                    ┌──────┴──┐ ┌──┴──────┐
                    │Decoder  │ │CT Head  │
                    └─────────┘ └─────────┘
                    x̂ (n_genes)  ŷ (n_ct)

Three training objectives:
  1. Reconstruction — MSE(x̂, x)          [preserves transcriptional variation]
  2. Cell-type      — CrossEntropy(ŷ, y)  [preserves biological structure]
  3. Neighbor       — optional triplet /  [preserves local neighbourhood]
                      contrastive loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selector import ConcreteSelector, STGSelector


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(dims: list[int], dropout: float = 0.1, batch_norm: bool = True) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:         # no BN / dropout on final layer
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class GeneCompressionAE(nn.Module):
    """End-to-end gene panel discovery + reconstruction autoencoder.

    Args:
        n_genes:        Number of input genes (post-HVG filtering).
        n_selected:     Panel size (k).
        n_celltypes:    Number of cell type classes for the auxiliary head.
        latent_dim:     Dimension of the bottleneck latent space.
        encoder_dims:   Hidden layer sizes for the encoder MLP.
        decoder_dims:   Hidden layer sizes for the decoder MLP.
        selector_type:  ``"concrete"`` (default) or ``"stg"``.
        temperature:    Initial temperature for ConcreteSelector.
        dropout:        Dropout rate in encoder/decoder.
    """

    def __init__(
        self,
        n_genes: int,
        n_selected: int,
        n_celltypes: int,
        latent_dim: int = 64,
        encoder_dims: list[int] | None = None,
        decoder_dims: list[int] | None = None,
        selector_type: str = "concrete",
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_selected = n_selected
        self.n_celltypes = n_celltypes
        self.latent_dim = latent_dim

        # ── Gene selector ────────────────────────────────────────────────────
        if selector_type == "concrete":
            self.selector = ConcreteSelector(n_genes, n_selected, temperature)
            selector_out = n_selected
        elif selector_type == "stg":
            self.selector = STGSelector(n_genes, target_k=n_selected)
            selector_out = n_genes      # STG returns full-dim gated vector
        else:
            raise ValueError(f"Unknown selector_type: {selector_type!r}")
        self.selector_type = selector_type

        # ── Encoder MLP ──────────────────────────────────────────────────────
        enc_dims = encoder_dims or [256, 128]
        self.encoder = _mlp(
            [selector_out] + enc_dims + [latent_dim],
            dropout=dropout,
        )

        # ── Decoder MLP ──────────────────────────────────────────────────────
        dec_dims = decoder_dims or [128, 256]
        self.decoder = _mlp(
            [latent_dim] + dec_dims + [n_genes],
            dropout=dropout,
        )

        # ── Cell-type classification head ─────────────────────────────────────
        self.ct_head = nn.Linear(latent_dim, n_celltypes)

    # ------------------------------------------------------------------
    def set_temperature(self, temperature: float) -> None:
        """Update selector temperature (called by trainer during annealing)."""
        if self.selector_type == "concrete":
            self.selector.set_temperature(temperature)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run selector + encoder.

        Returns:
            h:     (batch, latent_dim) latent representation.
            W:     selection weight matrix (or gates for STG).
        """
        z_sel, W = self.selector(x)
        h = self.encoder(z_sel)
        return h, W

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns a dict with keys:
            ``x_hat``    reconstructed expression (batch, n_genes)
            ``h``        latent embedding        (batch, latent_dim)
            ``logits``   cell-type logits         (batch, n_celltypes)
            ``W``        selection matrix / gates
        """
        h, W = self.encode(x)
        x_hat = self.decoder(h)
        logits = self.ct_head(h)
        return {"x_hat": x_hat, "h": h, "logits": logits, "W": W}

    # ------------------------------------------------------------------
    def get_selected_genes(self, var_names: list[str] | None = None) -> list:
        """Return selected gene indices (or names if var_names provided)."""
        return self.selector.get_selected_genes(
            top_k=self.n_selected if self.selector_type == "stg" else None,
            var_names=var_names,
        )

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"n_genes={self.n_genes}, n_selected={self.n_selected}, "
                f"latent_dim={self.latent_dim}, selector={self.selector_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> GeneCompressionAE:
    """Build a GeneCompressionAE from a config dict.

    Example cfg::

        {
            "n_genes": 3000,
            "n_selected": 50,
            "n_celltypes": 25,
            "latent_dim": 64,
            "encoder_dims": [256, 128],
            "decoder_dims": [128, 256],
            "selector_type": "concrete",
            "temperature": 1.0,
            "dropout": 0.1,
        }
    """
    return GeneCompressionAE(
        n_genes=cfg["n_genes"],
        n_selected=cfg["n_selected"],
        n_celltypes=cfg["n_celltypes"],
        latent_dim=cfg.get("latent_dim", 64),
        encoder_dims=cfg.get("encoder_dims"),
        decoder_dims=cfg.get("decoder_dims"),
        selector_type=cfg.get("selector_type", "concrete"),
        temperature=cfg.get("temperature", 1.0),
        dropout=cfg.get("dropout", 0.1),
    )
