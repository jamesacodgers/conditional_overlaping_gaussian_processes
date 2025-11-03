import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf
from gpflow import default_float
from gpflow.base import Parameter, _cast_to_dtype

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
        self.seed = seed

    def create_functions(self, test_type):
        """creates the test functions. This is done by drawing samples from what is essentially a GP prior then
        fitting a smoothing spline to it to create callable functions.
        :returns fs: a list of the data generating functions"""

        # create a GP with one observed data point far away from the domain of interest

        lengthscales_X = self.lengthscales_X[:self.observed_dims]
        lengthscales_H = self.lengthscales_H[:self.latent_dims]

        kernel_X = gpflow.kernels.RBF(lengthscales=tf.convert_to_tensor(lengthscales_X,
                                                                        dtype=default_float()), variance=2,
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

                kernel = gpflow.kernels.RBF(lengthscales=tf.convert_to_tensor(lengthscales_X +lengthscales_H,
                                                                                dtype=default_float()), variance=1)
                X = np.array([[-1e6] * (self.observed_dims + self.latent_dims)])

            else:
                Exception('type must be one of unrelated, linear_relation or non-linear_relation')

        Y = np.array([[0.]])
        gp = gpflow.models.GPR(data=(tf.convert_to_tensor(X, dtype=default_float()),
                                     tf.convert_to_tensor(Y, dtype=default_float())), kernel=kernel)

        # take one extra sample as the first sample doesn't seem to interpolate very well for some reason

        if test_type == 'unrelated':
            samples = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), self.n_fun)

            functions = []
            for sample in samples:
                gp = gpflow.models.GPR(data=(tf.convert_to_tensor(self.x_full, dtype=default_float()),
                                         tf.convert_to_tensor(sample, dtype=default_float())), kernel=kernel_X)
                functions.append(gp)

        elif test_type == 'linear_relation':

            X = np.array([[-1e6] * (self.observed_dims)])
            x_new = self.x_full
            Y = np.array([[0.]])

            h_new = np.random.uniform(-1, 1, (self.n_fun, self.latent_dims))
            H_news = np.repeat(h_new, repeats=len(self.x_full), axis=0)
            self.h_new = h_new
            samples = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), self.latent_dims)

            if self.non_zero_kap:
                ValueError('non zero kappa not implemented')
                # samples2 = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), self.n_fun)
            # else:
            #     samples2 = tf.zeros([self.n_fun, len(x_new), 1], tf.float64)

            self.x_new = x_new
            self.latents = samples
            functions = []

            for i in range(self.n_fun):
                y = samples[0]*self.h_new[i, 0] + samples[1]*self.h_new[i, 1] #+ samples2[i]
                gp = gpflow.models.GPR(data=(tf.convert_to_tensor(self.x_full, dtype=default_float()),
                                             tf.convert_to_tensor(y,
                                                 dtype=default_float())), kernel=kernel_X)
                functions.append(gp)

        else:
            sample = gp.predict_f_samples(tf.convert_to_tensor(x_new, dtype=default_float()), 1)
            functions = []
            for i in range(self.n_fun):
                gp = gpflow.models.GPR(data=(tf.convert_to_tensor(
                    x_new[len(self.x_full) * i:len(self.x_full) * (i + 1), :self.observed_dims], dtype=default_float()),
                                                         tf.convert_to_tensor(
                                                             sample[0][len(self.x_full) * i:len(self.x_full) * (i + 1)],
                                                             dtype=default_float())), kernel=kernel_X)
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
            y_ = self.ys[i].numpy()[idx][:n_points[i]]

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
                plt.plot(self.x_full, ys_mean.numpy(), label=f'function {i + 1}', alpha=0.5)
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
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1], ys_mean.numpy().reshape(self.xs_full[1].shape).T,
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
                ax[i].plot(self.x_full, ys_mean.numpy(), label=f'function {i + 1}', alpha=0.5)
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
                contour = ax[i].contourf(self.xs_full[0], self.xs_full[1], ys_mean.numpy().reshape(self.xs_full[1].shape).T,
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






