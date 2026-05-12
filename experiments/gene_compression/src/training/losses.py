"""
Loss functions for the gene compression autoencoder.

Total loss:
    L = λ_recon * L_recon
      + λ_ct    * L_ct
      + λ_nb    * L_neighbor   (optional)
      + λ_stg   * L_stg        (only for STGSelector)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all genes, averaged over batch."""
    return F.mse_loss(x_hat, x)


def celltype_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy cell-type classification from the latent representation."""
    return F.cross_entropy(logits, labels)


def neighbor_preservation_loss(
    h: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Soft triplet loss to keep cells of the same type close in latent space.

    Samples anchor-positive-negative triplets within the batch using cell type
    labels.  Encourages same-type cells to be closer than different-type cells
    by at least `margin`.

    Args:
        h:       (batch, latent_dim) latent embeddings.
        labels:  (batch,) integer cell type labels.
        margin:  Triplet margin.

    Returns:
        Scalar triplet loss (0 if no valid triplets in batch).
    """
    dist = torch.cdist(h, h, p=2)          # (batch, batch) pairwise L2

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
    label_ne = ~label_eq

    loss = torch.tensor(0.0, device=h.device)
    n_triplets = 0

    for i in range(len(h)):
        pos_mask = label_eq[i].clone()
        pos_mask[i] = False                 # exclude self
        neg_mask = label_ne[i]

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # Hardest positive and hardest negative within batch
        d_pos = dist[i][pos_mask].max()
        d_neg = dist[i][neg_mask].min()

        loss = loss + F.relu(d_pos - d_neg + margin)
        n_triplets += 1

    if n_triplets > 0:
        loss = loss / n_triplets

    return loss


def stg_l0_loss(model: nn.Module) -> torch.Tensor:
    """L0 regularisation for STGSelector — differentiable gate sparsity."""
    from ..models.selector import STGSelector
    for module in model.modules():
        if isinstance(module, STGSelector):
            return module.l0_loss()
    return torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss with configurable weights
# ─────────────────────────────────────────────────────────────────────────────

class CompressionLoss(nn.Module):
    """Weighted combination of all compression objectives.

    Args:
        lambda_recon:     Weight for reconstruction MSE.
        lambda_ct:        Weight for cell-type classification CE.
        lambda_neighbor:  Weight for neighbor preservation triplet loss.
                          Set to 0 to disable (expensive on large batches).
        lambda_stg:       Weight for STG L0 regularisation.
        triplet_margin:   Margin for the triplet loss.
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_ct: float = 0.5,
        lambda_neighbor: float = 0.0,
        lambda_stg: float = 0.0,
        triplet_margin: float = 1.0,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_ct = lambda_ct
        self.lambda_neighbor = lambda_neighbor
        self.lambda_stg = lambda_stg
        self.triplet_margin = triplet_margin

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        x: torch.Tensor,
        ct_labels: torch.Tensor,
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and per-component breakdown.

        Args:
            outputs:   Dict returned by GeneCompressionAE.forward().
            x:         Original expression matrix (batch, n_genes).
            ct_labels: Cell type integer labels (batch,).
            model:     The full model (needed for STG L0 term).

        Returns:
            total_loss:   Scalar tensor.
            breakdown:    Dict of named float values for logging.
        """
        breakdown: dict[str, float] = {}

        # Reconstruction
        l_recon = reconstruction_loss(outputs["x_hat"], x)
        total = self.lambda_recon * l_recon
        breakdown["recon"] = l_recon.item()

        # Cell type classification
        l_ct = celltype_loss(outputs["logits"], ct_labels)
        total = total + self.lambda_ct * l_ct
        breakdown["ct"] = l_ct.item()

        # Neighbor preservation (optional, skip if weight is 0)
        if self.lambda_neighbor > 0:
            l_nb = neighbor_preservation_loss(
                outputs["h"], ct_labels, self.triplet_margin
            )
            total = total + self.lambda_neighbor * l_nb
            breakdown["neighbor"] = l_nb.item()

        # STG L0 regularisation
        if self.lambda_stg > 0 and model is not None:
            l_stg = stg_l0_loss(model)
            total = total + self.lambda_stg * l_stg
            breakdown["stg_l0"] = l_stg.item()

        breakdown["total"] = total.item()
        return total, breakdown
