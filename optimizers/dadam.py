import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.optim import Optimizer

from core import PhysicsConfig

@dataclass
class SynapseAdamState:
    """Optimizer state tracking harmonic adaptation and resonance growth."""
    exp_avg: Tensor                     # First-order moment estimate
    exp_avg_sq: Tensor                  # Second-order moment estimate
    resonance_pattern: Tensor           # Mathematical resonance structure
    running_grad_norm: Optional[Tensor] = None
    running_param_norm: Optional[Tensor] = None
    max_exp_avg_sq: Optional[Tensor] = None
    step_count: int = 0


class SynapseAdam(Optimizer):
    """
    Adaptive optimizer inspired by Adam, extended with harmonic resonance alignment.

    This optimizer models parameter evolution as a naturally guided growth process,
    combining adaptive moment tracking with resonant field interactions.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        noise_scale: float = 0.01,
        noise_decay: float = 0.995,
        resonance_factor: float = 0.1,
        clip_value: float = 100.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, noise_scale=noise_scale,
            noise_decay=noise_decay, resonance_factor=resonance_factor,
            clip_value=clip_value
        )
        super().__init__(params, defaults)

        # Fundamental reference constants (normalized harmonics)
        self.ref_constants = torch.tensor([
            0.618033988749895,  # φ (golden ratio)
            0.367879441171442,  # 1/e
            0.318309886183791,  # 1/π
            0.707106781186547,  # 1/√2
            0.693147180559945,  # ln(2)
            0.786151377757423   # √φ
        ])

        self.step_count = 0

    def _init_state(self, p: Tensor) -> SynapseAdamState:
        """Initialize the optimizer state for a parameter tensor."""
        return SynapseAdamState(
            exp_avg=torch.zeros_like(p),
            exp_avg_sq=torch.zeros_like(p),
            resonance_pattern=self._generate_resonance_pattern(p.shape),
            running_grad_norm=torch.zeros(1, device=p.device),
            running_param_norm=torch.zeros(1, device=p.device),
            max_exp_avg_sq=torch.zeros_like(p) if self.defaults['amsgrad'] else None
        )

    def _safe_norm(self, tensor: Tensor, eps: float = 1e-8) -> Tensor:
        """Numerically stable vector norm."""
        return torch.sqrt(torch.sum(tensor * tensor) + eps)

    def _generate_resonance_pattern(self, shape: tuple) -> Tensor:
        """Generate a resonance lattice using trigonometric harmonics."""
        coords = [torch.arange(s) for s in shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        phases = [(grid + 1) * math.pi / 6*]()
