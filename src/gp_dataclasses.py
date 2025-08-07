from dataclasses import dataclass, field
import torch
from typing import Optional, Literal


@dataclass
class InducingConfig:
    n_inducing: int                         # Number of inducing points (M)
    selection: Literal["perm", "kmeans"]    # Strategy for selecting Z
    seed: Optional[int] = None              # Random seed (for reproducibility)
    
@dataclass
class DatasetConfig:
    """Configuration for dataset selection and parameters."""
    type: Literal["oil", "swiss_roll"] = "oil"  # Dataset type
    n_samples: int = 1000                        # Number of samples (for synthetic datasets)
    noise: float = 0.1                           # Noise level (for synthetic datasets)
    random_state: Optional[int] = None           # Random seed for dataset generation
    
@dataclass
class InitXDistSsvi:
    """
    Used in train_gp_lvm_ssvi:
        - method: "default" | "custom"
        - custom_path: path to JSON, if method == "custom"
    """
    method: str = "default"
    custom_path: Optional[str] = None

@dataclass
class LR:
    x: float
    hyp: float
    alpha: float


@dataclass
class InnerIters:
    start: int
    after: int
    switch: int


@dataclass
class Training:
    batch_size: int
    total_iters: int
    inner_iters: InnerIters


@dataclass
class Rho:
    t0: float
    k: float


@dataclass
class GPSSVIConfig:
    device: str
    debug: bool
    jitter: float
    max_exp: float
    q_latent: int
    init_signal_to_noise_ratio: float
    num_u_samples_per_iter: int
    lr: LR = field(default_factory=lambda: LR(x=0.0, hyp=0.0, alpha=0.0))
    rho: Rho = field(default_factory=lambda: Rho(t0=0.0, k=0.0))
    training: Training = field(default_factory=lambda: Training(batch_size=0, total_iters=0, inner_iters=InnerIters(start=0, after=0, switch=0)))
    inducing: InducingConfig = field(default_factory=lambda: InducingConfig(n_inducing=0, selection="perm"))
    init_latent_dist: InitXDistSsvi = field(default_factory=lambda: InitXDistSsvi())
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())

    def device_resolved(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Config auto selected device:", self.device)
        return torch.device(self.device)