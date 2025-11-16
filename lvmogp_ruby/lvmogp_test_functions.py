import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d, RBFInterpolator


class RBFKernel:
    """RBF (Squared Exponential) kernel replacement for GPFlow."""
    def __init__(self, lengthscales, variance=1.0, active_dims=None):
        self.lengthscales = np.array(lengthscales) if not isinstance(lengthscales, np.ndarray) else lengthscales
        self.variance = variance
        self.active_dims = active_dims
        
    def __call__(self, X1, X2=None):
        """Compute kernel matrix K(X1, X2)."""
        if X2 is None:
            X2 = X1
            
        # Apply active dimensions if specified
        if self.active_dims is not None:
            X1 = X1[:, self.active_dims]
            X2 = X2[:, self.active_dims]
            
        # Compute squared distances scaled by lengthscales
        # ||x1 - x2||^2 / l^2
        X1_scaled = X1 / self.lengthscales
        X2_scaled = X2 / self.lengthscales
        
        # Efficient distance computation
        X1_sq = np.sum(X1_scaled**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_scaled**2, axis=1, keepdims=True)
        distances_sq = X1_sq + X2_sq.T - 2 * X1_scaled @ X2_scaled.T
        
        # RBF kernel: variance * exp(-0.5 * r^2)
        return self.variance * np.exp(-0.5 * distances_sq)


class SimpleGP:
    """Simple GP model for generating test functions (replacement for GPFlow GPR)."""
    def __init__(self, X, Y, kernel, noise_variance=1e-6):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.kernel = kernel
        self.noise_variance = noise_variance
        
        # Precompute for predictions
        self.K = kernel(self.X) + np.eye(len(self.X)) * noise_variance
        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y))
        
    def predict_f(self, X_new):
        """Predict mean and variance at new points."""
        X_new = np.array(X_new)
        K_s = self.kernel(self.X, X_new)
        K_ss = self.kernel(X_new)
        
        # Mean
        mean = K_s.T @ self.alpha
        
        # Variance
        v = np.linalg.solve(self.L, K_s)
        var = K_ss - v.T @ v
        
        return mean, np.maximum(np.diag(var).reshape(-1, 1), 1e-10)
    
    def predict_y(self, X_new):
        """Predict with observation noise."""
        mean, var = self.predict_f(X_new)
        var = var + self.noise_variance
        return mean, var
    
    @staticmethod
    def sample_prior(X, kernel, n_samples=1, seed=None):
        """Sample from GP prior."""
        if seed is not None:
            np.random.seed(seed)
            
        X = np.array(X)
        K = kernel(X) + np.eye(len(X)) * 1e-6  # Add jitter for numerical stability
        L = np.linalg.cholesky(K)
        
        # Generate samples
        samples = []
        for _ in range(n_samples):
            z = np.random.randn(len(X), 1)
            sample = L @ z
            samples.append(sample)
            
        return np.array(samples).squeeze()  # Shape: (n_samples, n_points, 1) or (n_points, 1)


class TestFun:

    def __init__(self, domain, seed, n_fun, test_type, observed_dims, latent_dims=2, max_points=100, noise=0.1, n_grid_points=100,
                 same_points=False, lengthscales_X=None, lengthscales_H=None, non_zero_kap=False):

        x_full = np.linspace(domain[0], domain[1], n_grid_points)

        if observed_dims == 1:
            self.x_full = x_full.reshape(len(x_full), 1)
        else:
            x_full_1, x_full_2 = np.meshgrid(x_full, x_full)
            X_full = np.hstack([np.expand_dims(x_full_1.ravel(), axis=1),
                                np.expand_dims(x_full_2.ravel(), axis=1)])
            self.x_full = X_full
            self.xs_full = (x_full_1, x_full_2)

        self.n_fun = n_fun
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.domain = domain
        self.max_points = max_points
        self.xs = []
        self.ys = []
        self.fun_no = []
        self.noise = noise
        self.h_new = None
        self.test_type = test_type
        self.non_zero_kap = non_zero_kap
        self.seed = seed  # Set seed before create_functions needs it

        if lengthscales_H is None:
            self.lengthscales_H = [0.2, 0.15]
        else:
            self.lengthscales_H = lengthscales_H
        if lengthscales_X is None:
            self.lengthscales_X = [0.1, 0.1]
        else:
            self.lengthscales_X = lengthscales_X

        self.functions = self.create_functions(test_type)

        if same_points:
            points = np.sort(np.random.uniform(self.domain[0], self.domain[1], (self.max_points, self.observed_dims)), axis=0)
            for i in range(self.n_fun):
                self.xs.append(points)
        else:
            for i in range(self.n_fun):
                self.xs.append(np.random.uniform(self.domain[0], self.domain[1], (self.max_points, self.observed_dims)))

        for i in range(self.n_fun):
            self.ys.append(self.function_with_noise(self.functions[i], self.xs[i], noise=self.noise))
            # self.fun_no.append(np.ones([len(self.max_points)])*i)

        self.y = None

    def create_functions(self, test_type):
        """creates the test functions. This is done by drawing samples from what is essentially a GP prior then
        fitting a smoothing spline to it to create callable functions.
        :returns fs: a list of the data generating functions"""

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # create a GP with one observed data point far away from the domain of interest
        lengthscales_X = self.lengthscales_X[:self.observed_dims]
        lengthscales_H = self.lengthscales_H[:self.latent_dims]

        kernel_X = RBFKernel(lengthscales=lengthscales_X, variance=2.0,
                            active_dims=list(range(self.observed_dims)))

        if test_type == 'unrelated' or test_type == 'linear_relation':
            kernel = kernel_X
            X = np.array([[-1e6] * (self.observed_dims)])
            x_new = self.x_full

        else:
            X = np.array([[-1e6] * (self.observed_dims + self.latent_dims)])

            h_new = np.random.uniform(-1, 1, (self.n_fun, self.latent_dims))
            H_news = np.repeat(h_new, repeats=len(self.x_full), axis=0)
            self.h_new = h_new
            x_new = self.get_x_new(self.x_full, self.n_fun, self.observed_dims, H_news)

            if test_type == 'non-linear_relation':
                kernel = RBFKernel(lengthscales=lengthscales_X + lengthscales_H, variance=1.0)
                X = np.array([[-1e6] * (self.observed_dims + self.latent_dims)])
            else:
                raise Exception('type must be one of unrelated, linear_relation or non-linear_relation')

        Y = np.array([[0.]])

        # Sample from GP prior
        if test_type == 'unrelated':
            # Sample n_fun independent functions
            samples = SimpleGP.sample_prior(x_new, kernel, n_samples=self.n_fun)
            if samples.ndim == 2:
                samples = samples[:, :, np.newaxis]  # Add output dimension

            functions = []
            for i in range(self.n_fun):
                sample = samples[i] if samples.shape[0] == self.n_fun else samples[:, i:i+1]
                gp = SimpleGP(X=self.x_full, Y=sample, kernel=kernel_X)
                functions.append(gp)

        elif test_type == 'linear_relation':
            X = np.array([[-1e6] * (self.observed_dims)])
            x_new = self.x_full
            Y = np.array([[0.]])

            h_new = np.random.uniform(-1, 1, (self.n_fun, self.latent_dims))
            H_news = np.repeat(h_new, repeats=len(self.x_full), axis=0)
            self.h_new = h_new
            
            # Sample latent functions
            samples = SimpleGP.sample_prior(x_new, kernel, n_samples=self.latent_dims)
            if samples.ndim == 2:
                samples = samples[:, :, np.newaxis]
            
            if self.non_zero_kap:
                raise ValueError('non zero kappa not implemented')

            self.x_new = x_new
            self.latents = samples
            functions = []

            for i in range(self.n_fun):
                # Linear combination of latent functions
                y = samples[0] * self.h_new[i, 0] + samples[1] * self.h_new[i, 1]
                gp = SimpleGP(X=self.x_full, Y=y, kernel=kernel_X)
                functions.append(gp)

        else:  # non-linear_relation
            sample = SimpleGP.sample_prior(x_new, kernel, n_samples=1)
            if sample.ndim == 1:
                sample = sample[:, np.newaxis]
            
            functions = []
            for i in range(self.n_fun):
                start_idx = len(self.x_full) * i
                end_idx = len(self.x_full) * (i + 1)
                X_i = x_new[start_idx:end_idx, :self.observed_dims]
                Y_i = sample[start_idx:end_idx]
                gp = SimpleGP(X=X_i, Y=Y_i, kernel=kernel_X)
                functions.append(gp)

        self.functions = functions
        return self.functions

    def get_x_new(self, x_full, n_fun, observed_dims, H_news):
        return np.concatenate([np.vstack([x_full]*n_fun).reshape(len(H_news), observed_dims), H_news], axis=1)


    def create_data(self, n_points, random_idx=None):
        """Create the data. This is done by randomly choosing input values then evaluating the functions at those  points
        with noise.

        Returns:
        _______
        X: numpy array
            inputs
        fun_no: numpy array
            function numbers
        y: numpy array
            output values
            """

        self.X = np.array([[]]).reshape(0, self.observed_dims)  # input values
        self.y = np.array([[]]).T  # output values
        self.fun_no = np.array([[]]).T  # number of the function that the point is observed on

        if type(n_points) is int:
            n_points = [n_points]*self.n_fun
        np.random.seed(self.seed)
        for i, fun in enumerate(self.functions):
            if random_idx is not None:
                idx = random_idx
            else:
                idx = range(0, len(self.xs[i]))
            x_ = self.xs[i][idx][:n_points[i]]
            # ys[i] is already numpy array
            y_ = self.ys[i][idx][:n_points[i]]

            fun_no_ = np.ones((len(x_), 1)) * i
            self.X = np.concatenate([self.X, x_])
            self.y = np.concatenate([self.y, y_])
            self.fun_no = np.concatenate([self.fun_no, fun_no_])

        return self.X, self.fun_no, self.y

    def function_with_noise(self, fun, x, noise):
        """one of the data generating functions
        Parameters
        __________
        fun: function
            function to which noise is to be added
        x: numpy array
            inputs values
        noise: float
            The standard deviation of the noise to be added to the function output"""
        mean, variance = fun.predict_y(x)
        return mean + np.random.normal(0, noise, (len(x), 1))

    def get_dataset(self, stdzr=None):
        """Create and store a dataset DataFrame (formerly Gumbi DataSet)."""
        x_columns = ['x1', 'x2', 'x3'][:self.observed_dims]

        df = pd.DataFrame(columns=x_columns + ['fun_no', 'y'])

        for i, column in enumerate(x_columns):
            df[column] = self.X[:, i].flatten()

        df['fun_no'] = [f'fun_{int(number[0])}' for number in self.fun_no]
        df['y'] = self.y.flatten()

        self.ds = df
        return self.ds


    def plot_data(self, n_points):

        if self.observed_dims == 1:
            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                plt.plot(self.x_full, ys_mean, label=f'function {i + 1}', alpha=0.5)
                idx = np.where(self.fun_no == i)
                x_ = self.xs[i][:n_points[i]]
                y_ = self.ys[i][:n_points[i]]
                plt.scatter(x_, y_, label=f'data  {i + 1}')

            plt.title('Data')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('x')
            plt.ylabel('y')
            plot_lims = [np.min(np.concatenate(self.ys))-0.25, np.max(np.concatenate(self.ys))+0.25]
            plt.ylim(plot_lims[0], plot_lims[1])
            return plot_lims

        if self.observed_dims == 2:

            fig, axs = plt.subplots(ncols=int(np.ceil(self.n_fun / 2)), nrows=2, figsize=(np.ceil(self.n_fun / 2) * 3, 5))
            ax = axs.flatten()

            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1], ys_mean.reshape(self.xs_full[1].shape).T,
                                         label=f'sample {i + 1}')

                # idx = np.where(self.fun_no == i)
                x_ = self.xs[i][:, 0]
                y_ = self.xs[i][:, 1]
                ax[i].scatter(x_, y_, label=f'data', color='k', marker='x')
                ax[i].set_title(f'function {i + 1}')
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                cbar = plt.colorbar(contour, ax=ax[i])

            plt.suptitle('Data')
            plt.tight_layout()
            return None

    def plot_data_seperate_plots(self):

        fig, axs = plt.subplots(ncols=self.n_fun, figsize=(3*self.n_fun, 3))

        ax = axs.flatten()

        if self.observed_dims == 1:
            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                ax[i].plot(self.x_full, ys_mean, label=f'function {i + 1}', alpha=0.5)
                idx = np.where(self.fun_no == i)
                x_ = self.xs[i]
                y_ = self.ys[i]
                # ax[i].scatter(x_, y_, label=f'data  {i + 1}')
                ax[i].set_xlabel('x')
                ax[i].set_ylabel('y')


            plt.suptitle('Data')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            plot_lims = [np.min(np.concatenate(self.ys))-0.25, np.max(np.concatenate(self.ys))+0.25]
            plt.ylim(plot_lims[0], plot_lims[1])
            return plot_lims

        if self.observed_dims == 2:

            fig, axs = plt.subplots(ncols=int(np.ceil(self.n_fun / 2)), nrows=2, figsize=(np.ceil(self.n_fun / 2) * 3, 5))
            ax = axs.flatten()

            for i, fun in enumerate(self.functions):
                ys_mean, ys_var = fun.predict_y(self.x_full)
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1], ys_mean.reshape(self.xs_full[1].shape).T,
                                         label=f'sample {i + 1}')

                # idx = np.where(self.fun_no == i)
                x_ = self.xs[i][:, 0]
                y_ = self.xs[i][:, 1]
                ax[i].scatter(x_, y_, label=f'data', color='k', marker='x')
                ax[i].set_title(f'function {i + 1}')
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                cbar = plt.colorbar(contour, ax=ax[i])

            plt.suptitle('Data')
            plt.tight_layout()
            return None


    def plot_hs(self):

        if self.h_new is not None:
            fig = plt.figure(figsize=(4, 4))
            axsH = plt.gca()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 3
            for i, H_coord in enumerate(self.h_new):
                axsH.scatter(H_coord[0], H_coord[1], label=f'{i + 1}', color=colors[i])
                axsH.annotate(f'{i + 1}', (H_coord[0], H_coord[1]))
                axsH.set_title(f"Test Function Latent Coordinates")
                axsH.set_xlabel(f'latent dimension 1')
                axsH.set_ylabel(f'latent dimension 2')
            plt.tight_layout()






