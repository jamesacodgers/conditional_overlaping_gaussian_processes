import torch
import types

# Local wrapper to mirror examples/toy_GPLVM.Struc_GPLVM without importing that file (which has side effects)
from src.struc_models import Struc_WS_GPLVM
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF


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
    # We will use each model's internal Sigma_x (default exp(0)=1) to mirror toy_GPLVM

    # Instantiate both models with identical hyperparameters
    struc = Struc_GPLVM(Vx=Vx, M=M, mu_x=mu_x_base, sigma2=1.0, P=P, trainable_V=False)
    ssvi = GPLVM_SSVI_AM_INF(Vx=Vx, M=M, mu_x=mu_x_base, sigma2=1.0, P=P, trainable_V=False, device="cpu")

    # Synchronize q(u) parameters so KL_u matches exactly
    with torch.no_grad():
        ssvi.gp._mu_u.copy_(struc.gp._mu_u)
        ssvi.gp._raw_chol.copy_(struc.gp._raw_chol)

    # Generate identical inputs for both ELBOs using the same random base for q(u)
    # Draw U from Struc sampling and feed the SAME fixed samples to both models
    U_samples = struc.get_u_samples().detach()  # [P, M, C, L]
    mu_x_P = struc.mu_x_train.detach()          # [P, N, Q]
    Sigma_x_P = struc.Sigma_x_train.detach()    # [P, N, Q]

    # Force Struc to use the SAME fixed U_samples under the hood
    U_vec_fixed = U_samples.reshape(U_samples.shape[0], U_samples.shape[1], -1)
    struc.get_u_samples = types.MethodType(lambda self: U_samples, struc)
    struc.get_u_vec_samples = types.MethodType(lambda self: U_vec_fixed, struc)

    # ELBO via Struc (uses fixed U_samples now)
    elbo_struc = struc.gplvm_elbo(y)

    # ELBO via SSVI with explicit inputs
    elbo_ssvi = ssvi.gplvm_elbo(y=y, mu_x=mu_x_P, Sigma_x=Sigma_x_P, U_samples=U_samples, batch_size=y.shape[0], total_size=y.shape[0])

    print("ELBO (Struc_GPLVM):", elbo_struc.item())
    print("ELBO (GPLVM_SSVI_AM_INF):", elbo_ssvi.item())
    print("Abs diff:", abs(elbo_struc.item() - elbo_ssvi.item()))

    # Optional assertion for equality within tolerance
    assert torch.allclose(elbo_struc, elbo_ssvi, atol=1e-6, rtol=1e-6), "ELBOs differ beyond tolerance"


if __name__ == "__main__":
    main()


