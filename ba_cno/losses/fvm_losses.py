from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FVMCompositeLoss(nn.Module):
    """Composite loss for direct multi-horizon FVM-style velocity prediction."""

    def __init__(
        self,
        output_steps: int = 5,
        delta: float = 1.0,
        boundary_weight: float = 0.1,
        flux_weight: float = 0.02,
        late_horizon_bonus: float = 0.4,
        loss_type: str = "huber",
    ) -> None:
        super().__init__()
        self.output_steps = int(output_steps)
        self.delta = float(delta)
        self.boundary_weight = float(boundary_weight)
        self.flux_weight = float(flux_weight)
        self.loss_type = str(loss_type).lower()

        horizon_weights = torch.linspace(1.0, 1.0 + float(late_horizon_bonus), steps=self.output_steps)
        self.register_buffer("horizon_weights", horizon_weights / horizon_weights.mean())

    def _velocity_loss(self, pred_velocity: torch.Tensor, true_velocity: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            loss = torch.abs(pred_velocity - true_velocity)
        else:
            loss = F.huber_loss(
                pred_velocity,
                true_velocity,
                reduction="none",
                delta=self.delta,
            )

        weights = self.horizon_weights.view(1, self.output_steps, 1).to(loss)
        return (loss * weights).mean()

    def _boundary_loss(
        self,
        pred_velocity: torch.Tensor,
        airfoil_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if airfoil_mask is None:
            return pred_velocity.new_zeros(())

        airfoil_mask = airfoil_mask.to(device=pred_velocity.device, dtype=torch.bool)
        if not torch.any(airfoil_mask):
            return pred_velocity.new_zeros(())

        return pred_velocity[airfoil_mask].pow(2).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        true_velocity: torch.Tensor,
        airfoil_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_velocity = outputs["velocity"]
        velocity_loss = self._velocity_loss(pred_velocity, true_velocity)
        boundary_loss = self._boundary_loss(pred_velocity, airfoil_mask)

        aux_losses = outputs.get("aux_losses", {})
        flux_loss = pred_velocity.new_zeros(())
        if isinstance(aux_losses, dict) and "flux_antisymmetry" in aux_losses:
            flux_loss = aux_losses["flux_antisymmetry"]

        total_loss = velocity_loss + self.boundary_weight * boundary_loss + self.flux_weight * flux_loss
        return {
            "loss": total_loss,
            "velocity_loss": velocity_loss,
            "boundary_loss": boundary_loss,
            "flux_loss": flux_loss,
        }
