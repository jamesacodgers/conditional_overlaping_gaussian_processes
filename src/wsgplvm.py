# Class containing the MOGPLVM logic

from typing import Optional, List
import torch
from torch import nn

from src.inner_optimisers import MultivariateNormalSolver
from src.eiot import generate_regular_simplex

class WSGPLVM(nn.Module):
    def __init__(
        self,
        sigma2: torch.Tensor,
        sigma2_s: torch.Tensor,
        beta: torch.Tensor,
        v_x: torch.Tensor,
        gamma: torch.Tensor = None,
        v_l: torch.Tensor = None,
        device: torch.device = torch.device('cpu')
        # TODO: play with different hyperparamters for each component
    ):
        torch.set_default_dtype(torch.float64)
        super(WSGPLVM, self).__init__()

        self._log_beta = nn.Parameter(torch.log(beta.to(device)))
        if gamma is not None:
            self._log_gamma = nn.Parameter(torch.log(gamma.to(device)))
        else:
            assert v_l is None
            self._log_gamma = None
        
        self._log_sigma2 = nn.Parameter(torch.log(sigma2.to(device)))
        self._log_sigma2_s = nn.Parameter(torch.log(sigma2_s.to(device)))

        self.v_x = nn.Parameter(v_x.to(device))
        if v_l is not None:
            self.v_l = nn.Parameter(v_l.to(device))
        else:
            assert gamma is None
            self.v_l = None

        self.device = device

    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test=None):
        A_train = self.get_A_train() # C_unobs x C_unobs
        b_train = self.get_b_train(D_train, R_train) # N x C_unobs
        const_train = self.get_const_train(D_train, R_train)
        training_inner_elbo = self.inner_train_solver(A_train, b_train) + const_train
        kl = self.get_kl()
        if D_test is None: 
            return training_inner_elbo - kl
        else: 
            A_test = self.get_A_test()
            b_test = self.get_b_test(D_test)
            const_test = self.get_const_test(D_test)
            test_inner_elbo = self.inner_test_solver(A_test, b_test) + const_test
            return training_inner_elbo + test_inner_elbo - kl

    def ard_kernel(self, x1, x2, length_scale, variance=1) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        # TODO: Correct for multiple kernels
        # wavelength_kernel
        # latent_kernel
        L2_diffs = weighted_L2_diff(x1, x2, 1 / length_scale)
        return variance * torch.exp(-1 / 2 * L2_diffs)

    def get_K_vv(self) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        K_vxvx = self.get_K_vxvx()
        if self.v_l is None:
            return self.sigma2_s * K_vxvx + 1e-6 * torch.eye(self.num_inducing_points, device=self.device)
        K_vlvl = self.get_K_vlvl()
        K_vv_expanded = (
            K_vxvx[
                :,
                :,
                None,
                None,
            ]
            * K_vlvl[
                None,
                None,
                :,
                :,
            ]
        )
        return self.sigma2_s * tile_final_dims_to_matrix(
            K_vv_expanded
        ) + 1e-6 * torch.eye(self.num_inducing_points, device=self.device)

    def get_K_vxvx(self) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        return self.ard_kernel(
            self.v_x, self.v_x, self.beta
        )  # + 1e-5*torch.eye(self.v_x.shape[0])

    def get_K_vlvl(self) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        return self.ard_kernel(
            self.v_l, self.v_l, self.gamma
        )  # + 1e-5*torch.eye(self.v_l.shape[0])

    def get_xi_0(self, datasets: List[Dataset]) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        xi_0 = torch.zeros(1).to(self.device)
        for dataset in datasets:
            xi_0 += self._get_xi_0_single(dataset)
        return xi_0

    def get_xi_1(self, datasets: List[Dataset]) -> torch.Tensor:
        if self.v_l is None:
            xi_1 = torch.zeros(
                datasets[0].num_measurements_per_data_point,
                self.num_inducing_points * datasets[0].num_components, device=self.device)
        else:
            xi_1 = torch.zeros(self.num_inducing_points * datasets[0].num_components, device=self.device)
        for dataset in datasets:
            xi_1 += self._get_xi_1_single(dataset)
        return xi_1

    def get_xi_2(self, datasets: List[Dataset]) -> torch.Tensor:
        L = self.num_inducing_points
        C = datasets[0].num_components
        xi_2 = torch.zeros(L * C, L * C, device =self.device)
        for dataset in datasets:
            xi_2 += self._get_xi_2_single(dataset)
        return xi_2

    def _get_xi_0_single(self, dataset: Dataset) -> torch.Tensor:
        r_outer = dataset.get_r_outer()
        if self.v_l is None:
            return self.sigma2_s * (r_outer.diagonal(dim1=-2, dim2=-1).sum())
        return (
            dataset.observations.num_measurements_per_datapoint
            * self.sigma2_s
            * (r_outer.diagonal(dim1=-2, dim2=-1).sum())
        )

    def _get_xi_1_single(self, dataset: Dataset) -> torch.Tensor:
        # TODO: correct for multiple kernels
        # Note: this is implimented only for the case where all kernels are the same
        psi1 = self.get_psi_1(dataset)  # NxMxL
        R = dataset.components_distribution.get_r()
        if self.v_l is None:
            expanded = (
                dataset.observations.get_observations()[:, :, None, None]
                * R[:, None, :, None]
                * psi1[:, None, None, :]
            )  # N x M x C x L
            return expanded.sum(axis=0).reshape(
                dataset.num_measurements_per_data_point, -1
            )
        expanded = (
            dataset.observations.get_observations()[:, :, None, None]
            * R[:, None, :, None]
            * psi1[:, :, None, :]
        )  # N x M x C x L
        return expanded.sum(axis=[0, 1]).flatten()

    def get_psi_1(self, dataset: Dataset) -> torch.Tensor:
        # TODO: Correct for multiple kernles
        psi_1_x = self.get_psi_1_x(dataset.mu_x, dataset.Sigma_x)  # N x L_x
        if self.v_l is None:
            return self.sigma2_s * psi_1_x
        K_lvl = self.get_K_lvl(dataset.observations.get_inputs())  # M x L_l
        psi_1_expanded = (
            self.sigma2_s * psi_1_x[:, None, :, None] * K_lvl[None, :, None, :]
        )
        return psi_1_expanded.reshape(
            dataset.num_data_points,
            dataset.observations.num_measurements_per_datapoint,
            -1,
        )

    def get_K_lvl(self, inputs: Optional[torch.Tensor]) -> torch.Tensor:
        return self.ard_kernel(inputs, self.v_l, self.gamma)  # M x L_l

    def get_psi_1_x(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        # TODO: Correct for multiple kernels
        scales = self.beta[None, :] + Sigma_x  # N x A
        det_beta = self.beta.prod()
        det_scales = scales.prod(axis=-1)
        differences = diff_between_rank_2_tensors(mu_x, self.v_x)  # N x L_x x A
        L2_scaled_differences = (differences**2 / scales[:, None, :]).sum(
            axis=-1
        )  # N x L_x
        return torch.sqrt(det_beta / det_scales[:, None]) * torch.exp(
            -1 / 2 * L2_scaled_differences
        )  # N x L_x

    def _get_xi_2_single(self, dataset: Dataset) -> torch.Tensor:
        # TODO: correct for multiple kernels
        # print("start psi 2")
        r_outer_product = dataset.get_r_outer()  # N x C x C
        psi_2_x = self.get_psi_2_x(dataset.mu_x, dataset.Sigma_x)  # N x L_x x L_x
        if self.v_l is not None:
            K_lvl = self.get_K_lvl(dataset.observations.get_inputs())  # M x L_l
            # print("wavelength kernel outer product")
            wavelength_kernel_outer_product = (
                K_lvl[:, None, :] * K_lvl[:, :, None]
            )  # M x L_x x L_l
            wavelength_kernel_outer_product = wavelength_kernel_outer_product.sum(
                axis=0
            )  # L_l x L_l
            # print("get psi x")
            kernel_expectation_expanded = (
                wavelength_kernel_outer_product[None, None, None, :, :]
                * psi_2_x[:, :, :, None, None]
            )  # N x L_x x L_x x L_l x L_l
            # print("tile")
            kernel_expectation = tile_final_dims_to_matrix(kernel_expectation_expanded)
        else:
            kernel_expectation = psi_2_x
        xi_2_expanded = (
            r_outer_product[:, :, :, None, None]
            * kernel_expectation[:, None, None, :, :]
        )
        # print("do expansion")
        # print("sum")
        helper_ones = torch.ones(xi_2_expanded.shape[0], device=self.device)
        psi_2 = xi_2_expanded.permute(1, 2, 3, 4, 0) @ helper_ones
        # print("tile")
        psi_2 = tile_final_dims_to_matrix(psi_2)
        return self.sigma2_s**2 * psi_2

    def get_psi_2_x(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        # TODO: correct for multiple kernels
        # Note - this is currently implimented such that all GPs for every component come from the same draw. To correct this there should be a additional C x C dimensions added so
        # print("get_psi x start ")
        det_beta = self.beta.prod()
        scales = self.beta[None, :] + 2 * Sigma_x
        det_scaled_covariance = scales.prod(axis=1)
        # print("diff between inducing")
        exponential_diff_between_inducing_term = torch.exp(
            -1 / 4 * weighted_L2_diff(self.v_x, self.v_x, 1 / self.beta)
        )  # L x L -> L_x x L_x
        # print("find average inducing")
        average_of_inducing_points = (self.v_x[:, None, :] + self.v_x[None, :, :]) / 2
        # print("diffs to average")
        helper_ones = torch.ones(self.num_latent_dims, device=self.device)
        diff_mean_to_average_inducing = (
            (mu_x[:, None, None, :] - average_of_inducing_points[None, :, :, :]) ** 2
            / scales[:, None, None, :]
        ) @ helper_ones
        # print("epx")
        exponential_diff_mean_to_average_inducing = torch.exp(
            -diff_mean_to_average_inducing
        )  # N x L_x x L_x
        return (
            torch.sqrt(det_beta / det_scaled_covariance)[:, None, None]
            * exponential_diff_between_inducing_term[None, :, :]
            * exponential_diff_mean_to_average_inducing
        )

    def get_sample_mean(
        self, dataset: Dataset, conditional_datasets: List[Dataset]
    ) -> torch.Tensor:
        K_xvx_c = self.ard_kernel(dataset.mu_x, self.v_x, self.beta)
        K_vxvx_c = self.get_K_vxvx()
        K_vxvx_inv_c = invert_covariance_matrix(K_vxvx_c)
        K_xvx = torch.block_diag(*[K_xvx_c] * dataset.num_components)
        K_vxvx_inv = torch.block_diag(*[K_vxvx_inv_c] * dataset.num_components)
        mu_u = self.get_mu_u(conditional_datasets)
        if self.v_l is not None:
            K_lvl = self.get_K_lvl(dataset.observations.get_inputs())
            K_vlvl = self.get_K_vlvl()
            K_vlvl_inv = invert_covariance_matrix(K_vlvl)
            kron_prod_x = K_xvx @ K_vxvx_inv
            kron_prod_l = K_lvl @ K_vlvl_inv
            K = tile_final_dims_to_matrix(
                kron_prod_x[:, :, None, None] * kron_prod_l[None, None, :, :]
            )
            return (
                (K @ mu_u)
                .reshape(
                    dataset.num_components,
                    dataset.num_data_points,
                    dataset.num_measurements_per_data_point,
                )
                .permute(1, 2, 0)
            )
        return (
            (K_xvx @ K_vxvx_inv @ mu_u)
            .reshape(
                dataset.num_measurements_per_data_point,
                dataset.num_components,
                dataset.num_data_points,
            )
            .permute(2, 0, 1)
        )

    def get_point_var(
        self, latent_variable: float, datasets: List[Dataset]
    ) -> torch.Tensor:
        if self.v_l is None:
            raise NotImplementedError(":(")
        input = make_grid(latent_variable, datasets[0].observations.get_inputs())
        K_xlxl_c = self.ard_kernel(input, input, torch.hstack([self.beta, self.gamma]), self.sigma2_s)
        K_xlxl = torch.block_diag(*[K_xlxl_c] * datasets[0].num_components)
        K_xlv_c = self.ard_kernel(input, self.v, torch.hstack([self.beta, self.gamma]), self.sigma2_s)
        K_xlv = torch.block_diag(*[K_xlv_c] * datasets[0].num_components)
        K_vv_c = self.get_K_vv()
        K_vv = torch.block_diag(*[K_vv_c] * datasets[0].num_components)
        K_vv_c_inv = invert_covariance_matrix(K_vv_c)
        K_vv_inv = torch.block_diag(*[K_vv_c_inv] * datasets[0].num_components)
        Sigma_u = self.get_Sigma_u(datasets)
       
        return torch.diag(K_xlxl - K_xlv @ (K_vv_inv -  K_vv_inv@ Sigma_u  @K_vv_inv )@ K_xlv.T).reshape(datasets[0].num_components, -1)

    def get_point_mean(self, latent_variable, datasets):
        if self.v_l is None:
            raise NotImplementedError(":(")
        input = make_grid(latent_variable, datasets[0].observations.get_inputs())
        K_xlv_c = self.ard_kernel(
            input, self.v, torch.hstack([self.beta, self.gamma]), self.sigma2_s
        )
        K_xlv = torch.block_diag(*[K_xlv_c] * datasets[0].num_components)
        K_vv = self.get_K_vv()
        K_vv_c_inv = invert_covariance_matrix(K_vv)
        K_vv_inv = torch.block_diag(*[K_vv_c_inv] * datasets[0].num_components)
        return (K_xlv @ K_vv_inv @ self.get_mu_u(datasets)).reshape(datasets[0].num_components, -1)

    def get_mu_u(self, datasets: List[Dataset]) -> torch.Tensor:
        xi_1 = self.get_xi_1(datasets)
        xi_2 = self.get_xi_2(datasets)
        K_c_vv = self.get_K_vv()
        K_vv = torch.block_diag(*[K_c_vv] * datasets[0].num_components)
        A = 1 / self.sigma2 * xi_2 + K_vv
        A_inv = invert_covariance_matrix(A)
        if self.v_l is None:
            mu_u = (
                1
                / self.sigma2
                * K_vv[None, :, :]
                @ A_inv[None, :, :]
                @ xi_1[:, :, None]
            )
            return mu_u
        mu_u = 1 / self.sigma2 * K_vv @ A_inv @ xi_1
        return mu_u

    def get_Sigma_u(self, datasets: List[Dataset]) -> torch.Tensor:
        xi_2 = self.get_xi_2(datasets)
        K_c_vv = self.get_K_vv()
        K_vv = torch.block_diag(*[K_c_vv] * datasets[0].num_components)
        A = 1 / self.sigma2 * xi_2 + K_vv
        A_inv = invert_covariance_matrix(A)
        return K_vv @ A_inv @ K_vv

    @property
    def beta(self):
        return torch.exp(self._log_beta)

    @property
    def gamma(self):
        if self._log_gamma is None:
            return None
        return torch.exp(self._log_gamma)

    @property
    def sigma2(self):
        return torch.exp(self._log_sigma2) + 1e-5  # This is to limit the minimum noise

    @sigma2.setter
    def sigma2(self, value):
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.Tensor([value]).to(self.device)))

    @property
    def sigma2_s(self):
        return torch.exp(self._log_sigma2_s)

    @property
    def num_inducing_points(self):
        return int(self.v.shape[0])

    @property
    def num_inducing_points_in_observed_space(self):
        return int(self.v_l.shape[0])

    @property
    def num_inducing_points_in_latent_space(self):
        return int(self.v_x.shape[0])

    @property
    def num_latent_dims(self):
        return int(self.v_x.shape[1])

    @property
    def v(self):
        if self.v_l is None:
            return self.v_x
        return make_grid(self.v_x, self.v_l)

    def parameters_no_noise(self):
        if self.v_l is None:
            return [self._log_beta, self._log_sigma2_s, self.v_x]
        return [self._log_beta, self._log_gamma, self._log_sigma2_s, self.v_x, self.v_l]

    def ard_parameters(self):
        return [self._log_beta, self._log_gamma, self._log_sigma2_s]


class MFEIOT(torch.nn.Module):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False):
        super(MFEIOT, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_U = torch.nn.Parameter(torch.randn(M, self.C))
        self._raw_chol_U = torch.nn.Parameter(torch.eye(self.C))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        if sum_to_one:
            self.linear_map = torch.vstack([torch.zeros(self.C-1), torch.eye(self.C-1)])
            self.linear_map[:self.C_obs, :self.C_obs-1] = generate_regular_simplex(self.C_obs-1)
            self.linear_map[:-self.C_unobs, ]
            self.omega = torch.zeros(self.C)
            self.omega[:self.C_obs] = torch.ones(self.C_obs)/self.C_obs
            self.inner_test_solver = MultivariateNormalSolver(self.C -1)
        else:
            self.linear_map = None
            self.inner_test_solver = MultivariateNormalSolver(self.C)
        self._log_priors_T = torch.nn.Parameter(torch.zeros(self.C_unobs))


    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test=None):
        s_expectation = self.get_s_expectation()
        s_obs_expectation = s_expectation[:,:self.C_obs]
        s_unobs_expectation = s_expectation[:,self.C_obs:]
        s_outer_expectation = self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T
        s_obs_outer_expectation = s_outer_expectation[:self.C_obs,:self.C_obs]
        s_cross_outer_expectation = s_outer_expectation[:self.C_obs,self.C_obs:]
        s_unobs_outer_expectation = s_outer_expectation[self.C_obs:,self.C_obs:]
        A_train = self.get_A_train(s_unobs_outer_expectation) # C_unobs x C_unobs
        b_train = self.get_b_train(D_train, R_train, s_unobs_expectation, s_cross_outer_expectation) # N x C_unobs
        const_train = self.get_const_train(D_train, R_train, s_obs_expectation, s_obs_outer_expectation)
        training_inner_elbo = self.inner_train_solver(A_train, b_train) + const_train
        kl = self.get_kl()
        if D_test is None: 
            return training_inner_elbo - kl
        else: 
            A_test = self.get_A_test(s_outer_expectation)
            b_test = self.get_b_test(D_test, s_expectation)
            const_test = self.get_const_test(D_test, s_expectation, s_outer_expectation)
            test_inner_elbo = self.inner_test_solver(A_test, b_test) + const_test
            return training_inner_elbo + test_inner_elbo - kl
        

    def get_s_expectation(self):
        ...

    def get_s_outer_expectation(self):
        ...

    def get_A_test(self, s_train_outer_expectation):
        if self.linear_map is not None:
            return self.linear_map.T@(1/(self.sigma2)*(s_train_outer_expectation))@self.linear_map # (C-1) x (C-1) 
        else:
            return 1/(self.sigma2)*(s_train_outer_expectation) # C x C 


    def get_b_test(self, D_test, s_expectation: torch.Tensor):
        if self.linear_map is not None: 
            return 1/(self.sigma2)*((D_test - self.omega[None, :]@s_expectation.T)[:, :, None, None]*s_expectation[None,:,:,None]*self.linear_map[None,None,:,:]).sum(dim=[1,2]) # N x (C-1)
        else:
            return 1/(self.sigma2)*D_test@s_expectation # N x C 
                
    def get_const_test(self, D_test, s_expectation: torch.Tensor, s_outer_expectation: torch.Tensor):
        if self.linear_map is not None:
            return -(self.M * D_test.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_test**2).sum() - 2 * (D_test[:, :, None] * s_expectation[None, :, :] *self.omega[None,None,:]).sum()  + D_test.shape[0]*(self.omega[None,:]*s_outer_expectation*self.omega[:, None]).sum())
        else: 
            return -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
        
    def get_A_train(self, s_unobs_outer_expectation):
        return 1/(self.sigma2)*(s_unobs_outer_expectation)
    
    def get_b_train(self, D_train, R_train, s_unobs_expectation, s_outer_cross_expectation):
        return 1/(self.sigma2)*(D_train@s_unobs_expectation - R_train@s_outer_cross_expectation)
                
    def get_const_train(self, D_train, R_train, s_obs_expectation, s_obs_outer_expectation):
        return -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train**2).sum() - 2 * torch.einsum("nm, mc, cn", D_train, s_obs_expectation, R_train.T) + torch.einsum("nc, cd, dn",R_train,s_obs_outer_expectation,R_train.T))
    
    def get_unobs_train_weights(self, D_train, R_train):
        b_train = 1/(self.sigma2)*(D_train - R_train@self.mu_S.T)@self.mu_T # N x C 
        A_train = torch.linalg.inv(1/(self.sigma2)*(self.mu_T.T@ self.mu_T + self.M*self.Sigma_T)) # C x C 
        return self.inner_train_solver.get_weight_params(A_train, b_train)
    
    def predict_weights(self, D_test):
        if self.linear_map is not None:
            A_test = self.linear_map.T@(1/(self.sigma2)*(self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T))@self.linear_map # (C-1) x (C-1) 
            b_test = 1/(self.sigma2)*((D_test - self.omega[None, :]@self.mu_S_T.T)[:, :, None, None]*self.mu_S_T[None,:,:,None]*self.linear_map[None,None,:,:]).sum(dim=[1,2]) # N x (C-1)
            mu_rho, Sigma_rho = self.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.linear_map.T + self.omega
            Sigma_r = self.linear_map@Sigma_rho@self.linear_map.T + 1e-8
        else:
            A_test = 1/(self.sigma2)*(self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T) # C x C 
            b_test = 1/(self.sigma2)*D_test@self.mu_S_T # N x C 
            mu_r, Sigma_r = self.inner_test_solver.get_weight_params(A_test, b_test)
        return torch.distributions.MultivariateNormal(mu_r,Sigma_r, validate_args=False)
    
    def get_kl_u(self):
        ...
        # variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        # prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.prior_Sigma)
        # return torch.distributions.kl_divergence(variational_dist, prior).sum()
    
    def get_kl_x(self):
        ...
        # variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        # prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.prior_Sigma)
        # return torch.distributions.kl_divergence(variational_dist, prior).sum()

    
    
