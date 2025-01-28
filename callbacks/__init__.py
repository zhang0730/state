from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
from models import PerturbationModel

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

class PerturbationMagnitudeCallback(Callback):
    """
    Callback that tracks the average magnitude of perturbation effects by measuring
    the L2 norm of the difference between predicted perturbed state and control/basal state.
    """
    
    def __init__(self):
        super().__init__()
                    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: PerturbationModel,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            # basal_states = pl_module.encode_basal_expression(batch["basal"])
            # perturbed_states = pl_module.perturb(batch["pert"], batch["basal"])
            # magnitudes = torch.norm(perturbed_states - basal_states, p=2, dim=1)
            # pert_magnitude = magnitudes.mean()
            pl_module.log("val/perturbation_magnitude", 0.0)