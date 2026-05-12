"""
Training loop for GeneCompressionAE with temperature annealing.

Temperature schedule
--------------------
    τ(t) = max(τ_min, τ_start × exp(−anneal_rate × t))

where t is the current training step.  High τ → soft gene selection
(exploration), low τ → hard selection (exploitation).  The transition
from soft to hard is gradual so gradients flow meaningfully throughout.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import CompressionLoss


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScheduler:
    """Exponential temperature annealing for the Concrete selector.

    Args:
        t_start:      Initial temperature (default 1.0 — fully soft).
        t_min:        Minimum temperature (default 0.01 — nearly hard).
        anneal_rate:  Exponential decay rate per step.
        anneal_every: Anneal every N training steps.
    """

    def __init__(
        self,
        t_start: float = 1.0,
        t_min: float = 0.01,
        anneal_rate: float = 3e-4,
        anneal_every: int = 100,
    ):
        self.t_start = t_start
        self.t_min = t_min
        self.anneal_rate = anneal_rate
        self.anneal_every = anneal_every
        self._step = 0

    def step(self) -> float:
        """Advance scheduler by one step, return current temperature."""
        self._step += 1
        return self.get_temperature()

    def get_temperature(self) -> float:
        t = self.t_start * math.exp(-self.anneal_rate * self._step)
        return max(self.t_min, t)

    def should_anneal(self) -> bool:
        return self._step % self.anneal_every == 0


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Training orchestrator for GeneCompressionAE.

    Args:
        model:          GeneCompressionAE instance.
        loss_fn:        CompressionLoss instance.
        optimizer:      A PyTorch optimizer.
        scheduler:      Optional LR scheduler.
        temp_scheduler: TemperatureScheduler for Concrete selector.
        device:         ``"cuda"``, ``"mps"``, or ``"cpu"``.
        save_dir:       Directory for checkpoints and logs.
        log_every:      Print/log metrics every N batches.
        grad_clip:      Max norm for gradient clipping (0 = disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: CompressionLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        temp_scheduler: TemperatureScheduler | None = None,
        device: str = "cpu",
        save_dir: str = "./checkpoints",
        log_every: int = 100,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temp_scheduler = temp_scheduler or TemperatureScheduler()
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.grad_clip = grad_clip

        self.history: list[dict] = []
        self._global_step = 0

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one full epoch over the training DataLoader.

        The DataLoader should yield batches of dicts (or tuples) with keys:
            ``"x"``      — (batch, n_genes) expression tensor
            ``"ct"``     — (batch,) integer cell type label tensor

        Returns:
            Mean per-component losses over the epoch.
        """
        self.model.train()
        epoch_losses: dict[str, list[float]] = {}

        for batch in loader:
            x = batch["x"].to(self.device)
            ct = batch["ct"].to(self.device)

            # Forward
            outputs = self.model(x)
            loss, breakdown = self.loss_fn(outputs, x, ct, model=self.model)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Temperature annealing
            self._global_step += 1
            if self.temp_scheduler.should_anneal():
                tau = self.temp_scheduler.step()
                self.model.set_temperature(tau)

            # Accumulate losses
            for k, v in breakdown.items():
                epoch_losses.setdefault(k, []).append(v)

            if self._global_step % self.log_every == 0:
                self._log_step(breakdown)

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Run evaluation on a DataLoader (no gradients)."""
        self.model.eval()
        epoch_losses: dict[str, list[float]] = {}
        all_h, all_ct = [], []

        for batch in loader:
            x = batch["x"].to(self.device)
            ct = batch["ct"].to(self.device)

            outputs = self.model(x)
            _, breakdown = self.loss_fn(outputs, x, ct, model=self.model)

            for k, v in breakdown.items():
                epoch_losses.setdefault(k, []).append(v)

            all_h.append(outputs["h"].cpu())
            all_ct.append(ct.cpu())

        # Cell type classification accuracy from the latent head
        all_logits_list = []
        self.model.eval()
        for batch in loader:
            x = batch["x"].to(self.device)
            out = self.model(x)
            all_logits_list.append(out["logits"].cpu())
            break  # already gathered h above

        means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}

        # Accuracy
        h_cat = torch.cat(all_h)
        ct_cat = torch.cat(all_ct)
        # We need logits for the full eval set
        logits_cat = self._get_all_logits(loader)
        preds = logits_cat.argmax(dim=-1)
        means["ct_accuracy"] = (preds == ct_cat).float().mean().item()

        return means

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_all_logits(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        logits = []
        for batch in loader:
            x = batch["x"].to(self.device)
            out = self.model(x)
            logits.append(out["logits"].cpu())
        return torch.cat(logits)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stop_patience: int = 10,
    ) -> list[dict]:
        """Full training loop with early stopping on validation total loss.

        Returns:
            List of per-epoch metric dicts.
        """
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            tau = self.temp_scheduler.get_temperature()
            record = {
                "epoch": epoch,
                "temperature": tau,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time_s": time.time() - t0,
            }
            self.history.append(record)
            self._print_epoch(record)

            # Checkpointing
            val_loss = val_metrics["total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self.save_checkpoint("best.pt")
            else:
                patience_count += 1

            if early_stop_patience and patience_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {patience_count} epochs).")
                break

        self.save_checkpoint("last.pt")
        return self.history

    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "history": self.history,
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._global_step = ckpt.get("global_step", 0)
        self.history = ckpt.get("history", [])

    # ------------------------------------------------------------------
    def _log_step(self, breakdown: dict[str, float]) -> None:
        parts = [f"step={self._global_step:6d}",
                 f"τ={self.temp_scheduler.get_temperature():.4f}"]
        parts += [f"{k}={v:.4f}" for k, v in breakdown.items()]
        print("  ".join(parts))

    def _print_epoch(self, record: dict) -> None:
        train_total = record.get("train_total", float("nan"))
        val_total = record.get("val_total", float("nan"))
        val_acc = record.get("val_ct_accuracy", float("nan"))
        tau = record.get("temperature", float("nan"))
        t = record.get("epoch_time_s", 0)
        print(
            f"Epoch {record['epoch']:4d} | "
            f"train_loss={train_total:.4f}  val_loss={val_total:.4f}  "
            f"val_ct_acc={val_acc:.3f}  τ={tau:.4f}  ({t:.1f}s)"
        )
