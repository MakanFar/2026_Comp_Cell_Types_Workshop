"""
Differentiable gene selection modules.

Two approaches are implemented:
  1. ConcreteSelector  — Concrete distribution / Gumbel-softmax (Balin et al. 2019).
     Learns a k × n_genes weight matrix; each row is a soft categorical over genes.
     At inference the row argmax gives the selected gene index.
  2. STGSelector       — Stochastic Gates (Yamada et al. 2020).
     Learns one gate per gene; L0 regularization drives most gates to zero.
     Good when you want the sparsity pattern to emerge rather than be fixed to k.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Concrete / Gumbel-softmax selector
# ─────────────────────────────────────────────────────────────────────────────

class ConcreteSelector(nn.Module):
    """Differentiable top-k gene selection via the Concrete distribution.

    During **training** each of the k selector rows is a Gumbel-softmax
    distribution over all n_genes.  The selected representation is the
    weighted sum of input genes: ``z = W_soft @ x``.

    During **evaluation** (or when ``hard=True``) we take the argmax of each
    row and return a hard one-hot, giving exactly k (possibly overlapping)
    genes.  Duplicates are resolved by taking the unique set in rank order.

    Args:
        n_genes:    Number of input genes (after HVG filtering).
        n_selected: Panel size k (e.g. 10, 25, 50, 100).
        temperature: Initial Gumbel-softmax temperature τ.  Annealed externally
                     by calling ``set_temperature(τ)``.
    """

    def __init__(self, n_genes: int, n_selected: int, temperature: float = 1.0):
        super().__init__()
        self.n_genes = n_genes
        self.n_selected = n_selected

        # Learnable logits: shape (n_selected, n_genes)
        self.logits = nn.Parameter(torch.empty(n_selected, n_genes))
        nn.init.xavier_uniform_(self.logits)

        self.temperature = temperature

    # ------------------------------------------------------------------
    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _soft_selection_matrix(self) -> torch.Tensor:
        """Return a (n_selected, n_genes) soft selection matrix."""
        if self.training:
            # Add Gumbel noise for exploration
            gumbel = -torch.log(-torch.log(
                torch.empty_like(self.logits).uniform_(1e-20, 1.0)
            ))
            noisy_logits = (self.logits + gumbel) / self.temperature
        else:
            noisy_logits = self.logits / self.temperature
        return F.softmax(noisy_logits, dim=-1)          # (n_selected, n_genes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_genes) log-normalised expression matrix.

        Returns:
            z_sel:    (batch, n_selected) — selected (or soft-selected) features.
            W:        (n_selected, n_genes) — the selection weight matrix used
                      (soft during training, hard one-hot at eval).
        """
        W = self._soft_selection_matrix()               # (n_selected, n_genes)

        if not self.training:
            # Hard one-hot at inference
            indices = W.argmax(dim=-1)                  # (n_selected,)
            W = F.one_hot(indices, self.n_genes).float()

        z_sel = x @ W.T                                 # (batch, n_selected)
        return z_sel, W

    # ------------------------------------------------------------------
    def get_selected_genes(self, var_names: list[str] | None = None) -> list:
        """Return the indices (or gene names) of the currently selected genes.

        Duplicate selections are resolved by keeping the first occurrence in
        ranking order (by max logit).
        """
        with torch.no_grad():
            order = self.logits.max(dim=-1).values.argsort(descending=True)
            indices = self.logits.argmax(dim=-1)[order]
            # Remove duplicates while preserving order
            seen, unique_idx = set(), []
            for idx in indices.tolist():
                if idx not in seen:
                    seen.add(idx)
                    unique_idx.append(idx)
        if var_names is not None:
            return [var_names[i] for i in unique_idx[: self.n_selected]]
        return unique_idx[: self.n_selected]

    def extra_repr(self) -> str:
        return (f"n_genes={self.n_genes}, n_selected={self.n_selected}, "
                f"temperature={self.temperature:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STG — Stochastic Gates
# ─────────────────────────────────────────────────────────────────────────────

class STGSelector(nn.Module):
    """Gene selection via Stochastic Gates (Yamada et al. 2020, ICML).

    Each gene j has a gate µ_j ∈ ℝ.  The gate value during training is::

        z_j = µ_j + 0.5 * ε,   ε ~ N(0, 1)
        g_j = clamp(sigmoid(z_j), 0, 1)

    The L0 regularization term (number of active gates) is approximated as::

        R_L0 = Σ_j sigmoid(µ_j)

    which is differentiable w.r.t. µ_j.

    At inference, the hard gate is ``g_j = (µ_j > 0).float()``.

    Args:
        n_genes:      Number of input genes.
        lam:          L0 regularization weight (controls sparsity).
        target_k:     Soft target for number of selected genes (used to set lam
                      automatically if provided; otherwise lam is used directly).
    """

    def __init__(self, n_genes: int, lam: float = 0.1, target_k: int | None = None):
        super().__init__()
        self.n_genes = n_genes

        if target_k is not None:
            # Auto-set lam so that the initial expected number of active gates
            # is close to target_k
            self.lam = 1.0 / target_k
        else:
            self.lam = lam

        # Gate means: initialise near 0.5 so ~half start open
        self.mu = nn.Parameter(torch.zeros(n_genes))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_genes)

        Returns:
            x_gated:  (batch, n_genes) — gated expression.
            gates:    (n_genes,) — gate values (soft during training).
        """
        if self.training:
            eps = torch.randn_like(self.mu)
            z = self.mu + 0.5 * eps
            gates = torch.clamp(torch.sigmoid(z), 0.0, 1.0)
        else:
            gates = (self.mu > 0).float()

        x_gated = x * gates.unsqueeze(0)               # (batch, n_genes)
        return x_gated, gates

    # ------------------------------------------------------------------
    def l0_loss(self) -> torch.Tensor:
        """Differentiable approximation to the L0 norm (expected active gates)."""
        return torch.sigmoid(self.mu).sum()

    # ------------------------------------------------------------------
    def get_selected_genes(
        self,
        top_k: int | None = None,
        var_names: list[str] | None = None,
    ) -> list:
        """Return indices or names of selected genes.

        If ``top_k`` is given, return the top-k genes by gate value regardless
        of the hard threshold.  Otherwise return all genes with µ > 0.
        """
        with torch.no_grad():
            if top_k is not None:
                indices = self.mu.topk(top_k).indices.tolist()
            else:
                indices = (self.mu > 0).nonzero(as_tuple=True)[0].tolist()
        if var_names is not None:
            return [var_names[i] for i in indices]
        return indices

    def extra_repr(self) -> str:
        n_active = (self.mu > 0).sum().item()
        return f"n_genes={self.n_genes}, lam={self.lam:.4f}, n_active={n_active}"
