"""
lr_scheduler.py

Implements the Noam learning rate scheduler strategy.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler: warms up linearly, then decays inverse-proportionally.
    """
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1) -> None:
        self.model_dim = d_model
        self.warmup_duration = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        """Computes the Noam scaling coefficient."""
        current_step = max(1, self.last_epoch + 1)
        scale_factor = (self.model_dim ** -0.5) * min(
            current_step ** -0.5,
            current_step * (self.warmup_duration ** -1.5)
        )
        return scale_factor

    def get_lr(self) -> list[float]:
        """Calculates LR arrays for all tracked optimizer parameters."""
        step_scale = self._get_lr_scale()
        new_learning_rates = [
            initial_lr * step_scale for initial_lr in self.base_lrs
        ]
        return new_learning_rates

def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list[float]:
    """Helper method to observe the trajectory of the LR."""
    mock_layer = torch.nn.Linear(1, 1)
    mock_optim = optim.Adam(mock_layer.parameters(), lr=1.0)
    mock_sched = NoamScheduler(mock_optim, d_model=d_model, warmup_steps=warmup_steps)
    
    lr_trajectory = []
    for _ in range(total_steps):
        current_lr = mock_optim.param_groups[0]["lr"]
        lr_trajectory.append(current_lr)
        mock_optim.step()
        mock_sched.step()
        
    return lr_trajectory