import torch
from src.struc_models import Struc_WS_GPLVM
import numpy as np

from src.ssvi_models_am_inf_sep_opt import NormalCoordinatorCustom

class Struc_GPLVM_Custom_ELBO(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True):
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2,
                         weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)
        self.coordinator = NormalCoordinatorCustom(module=self, sum_to_one=False)

    @property
    def mu_x(self):
        return self.mu_x_train

    @property
    def Sigma_x(self):
        return self.Sigma_x_train

    # Copied elbo from ssvi model. The idea is that now when we have flexible input params, it would be easier to train test prediction.
    def elbo(self, D_train, R_train, D_test=None, mu_x_train=None, Sigma_x_train=None, U_samples: torch.Tensor = None):
        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x_train, Sigma_x_train)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_train, Sigma_x_train)
        assert U_samples is not None, "U_samples must be provided for ELBO computation"
        # Replaced _s_expectation and _s_outer_expectation with _s_expectation_given_U and _s_outer_expectation_given_U
        s_train_expectation = self._s_expectation_given_U(kernel_expectation=kernel_train_expectation,
                                                          U_samples=U_samples)
        s_train_outer_expectation = self._s_outer_expectation_given_U(
            kernel_outer_expectation=kernel_train_outer_expectation,
            U_samples=U_samples)
        if D_test is None:
            return self._elbo(
                s_train_expectation,
                s_train_outer_expectation,
                D_train,
                R_train,
                mu_x_train,
                Sigma_x_train,
            )
        else:
            kernel_test_expectation = self._get_lv_kernel_expectation(self.mu_x_test, self.Sigma_x_test)
            kernel_test_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_test, self.Sigma_x_test)
            s_test_expectation = self.gp.get_s_expectation(kernel_test_expectation)
            s_test_outer_expectation = self.gp.get_s_outer_expectation(kernel_test_outer_expectation)
            return self._elbo(
                s_train_expectation,
                s_train_outer_expectation,
                D_train,
                R_train,
                mu_x_train,
                Sigma_x_train,
                D_test=D_test,
                s_test_expectation=s_test_expectation,
                s_test_outer_expectation=s_test_outer_expectation,
            )

    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train, mu_x_train, Sigma_x_train,
              s_test_expectation=None, s_test_outer_expectation=None, D_test=None):
        # Same reason to override as in elbo - to have mu_x_train, Sigma_x_train as arguments
        return self.coordinator._elbo(
            s_train_expectation=s_train_expectation,
            s_train_outer_expectation=s_train_outer_expectation,
            D_train=D_train,
            R_train=R_train,
            mu_x_train=mu_x_train,
            Sigma_x_train=Sigma_x_train,
            s_test_expectation=s_test_expectation,
            s_test_outer_expectation=s_test_outer_expectation,
            D_test=D_test
        )

    def gplvm_elbo(self, y, mu_x, Sigma_x, U_samples, batch_size, total_size):
        R_train = torch.ones((y.shape[0], 1), device=y.device, dtype=y.dtype)
        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x, Sigma_x)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x, Sigma_x)

        s_train_expectation = self._s_expectation_given_U(kernel_expectation=kernel_train_expectation,
                                                          U_samples=U_samples)
        s_train_outer_expectation = self._s_outer_expectation_given_U(
            kernel_outer_expectation=kernel_train_outer_expectation,
            U_samples=U_samples)

        const_train = self.coordinator.get_const_train(y, R_train, s_train_expectation, s_train_outer_expectation)

        training_inner_elbo = const_train.mean() - self.get_kl_x_from(mu_x, Sigma_x)

        # Scale const_train for mini-batches to represent full dataset contribution
        if batch_size != total_size:
            training_inner_elbo = training_inner_elbo * (total_size / batch_size)

        return training_inner_elbo - self.gp.get_kl_u()