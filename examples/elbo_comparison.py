import torch

# Local wrapper to mirror examples/toy_GPLVM.Struc_GPLVM without importing that file (which has side effects)
from src.struc_models import Struc_WS_GPLVM
from src.ssvi_models_v2 import GPLVM_SSVI


class Struc_GPLVM(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1.0, beta=None, sigma2_s=None, P=1, trainable_V=True):
        super().__init__(
            P=P,
            V=Vx,
            M=M,
            C_obs=1,
            mu_x_train=mu_x,
            mu_x_test=None,
            C_unobs=0,
            sigma2=sigma2,
            weight_distribution="continuous",
            beta=beta,
            sigma2_s=sigma2_s,
            trainable_V=trainable_V,
        )

    @property
    def mu_x(self):
        return self.mu_x_train

    @property
    def Sigma_x(self):
        return self.Sigma_x_train

    def gplvm_elbo(self, y: torch.Tensor) -> torch.Tensor:
        R_train = torch.ones((y.shape[0], 1), device=y.device, dtype=y.dtype)
        return super().elbo(y, R_train)


def main():
    torch.manual_seed(0)

    # Toy setup
    N = 32
    Q = 1
    M = 2
    P = 20  # arbitrary; both implementations should now agree for any P

    # Data (any values are fine; this is just a single ELBO evaluation)
    x_true = torch.randn(N, Q)
    y = torch.hstack([
        x_true + 0.1 * torch.randn_like(x_true),
        x_true**3 + 0.1 * torch.randn_like(x_true),
    ])
    y = (y - y.mean(dim=0)) / y.std(dim=0)

    # Inducing inputs and initial variational x
    Vx = torch.linspace(-3, 3, 8).reshape(-1, Q)
    mu_x_base = x_true.clone()
    sigma_x_base = 0.5 * torch.ones_like(mu_x_base)  # variance (diag)

    # Instantiate both models with identical hyperparameters
    struc = Struc_GPLVM(Vx=Vx, M=M, mu_x=mu_x_base, sigma2=1.0, P=P, trainable_V=False)
    ssvi = GPLVM_SSVI(Vx=Vx, M=M, mu_x=mu_x_base, sigma2=1.0, P=P, trainable_V=False)

    # Synchronize q(u) parameters so KL_u matches exactly
    with torch.no_grad():
        ssvi.gp._mu_u.copy_(struc.gp._mu_u)
        ssvi.gp._raw_chol.copy_(struc.gp._raw_chol)

    # Set Struc model Sigma_x to our chosen values
    with torch.no_grad():
        struc._log_sigma_x_train.copy_(sigma_x_base.log()[None, ...].repeat(P, 1, 1))

    # Generate identical inputs for both ELBOs using the same random base for q(u)
    # Draw U from the same q(u) (as per Struc sampling) and feed to SSVI explicitly
    U_samples = struc.get_u_samples().detach()  # [P, M, C, L]
    mu_x_P = struc.mu_x_train.detach()          # [P, N, Q]
    Sigma_x_P = struc.Sigma_x_train.detach()    # [P, N, Q]

    # ELBO via Struc (internal sampling path)
    elbo_struc = struc.gplvm_elbo(y)

    # ELBO via SSVI with explicit inputs
    elbo_ssvi = ssvi.gplvm_elbo(y=y, mu_x_train=mu_x_P, Sigma_x_train=Sigma_x_P, U_samples=U_samples)

    print("ELBO (Struc_GPLVM):", elbo_struc.item())
    print("ELBO (GPLVM_SSVI):", elbo_ssvi.item())
    print("Abs diff:", abs(elbo_struc.item() - elbo_ssvi.item()))

    # Optional assertion for equality within tolerance
    assert torch.allclose(elbo_struc, elbo_ssvi, atol=1e-6, rtol=1e-6), "ELBOs differ beyond tolerance"


if __name__ == "__main__":
    main()


