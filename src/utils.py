import torch
import tqdm
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_preds(R_test, cond_preds: torch.distributions.Distribution = None, mf_preds = None, title=None) -> None: 
    with torch.no_grad():
        # samples = preds.sample((1000, )).reshape((1000*P, 150,  2))
        plt.figure()
        # plt.title(f"C = {C}")
        plt.xlabel("True weights")
        plt.ylabel("predicted weights")

        plt.plot(R_test[:,0], R_test[:,0])
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        if title:
            plt.title(title)
        if cond_preds: 
            means = cond_preds.mean[:,:,0]
            vars = cond_preds.covariance_matrix[:,:,0,0]
            cond_preds_1d = norm(cond_preds.mean[:,:,0].detach().numpy(), np.sqrt(cond_preds.covariance_matrix[:,:,0,0].detach().numpy()))
            err = np.array([get_percentile(cond_preds_1d, 0.5) - get_percentile(cond_preds_1d, 0.025), get_percentile(cond_preds_1d, 0.975) - get_percentile(cond_preds_1d, 0.5)])
            plt.errorbar(R_test[:,0], means.mean(dim=0), yerr=err, color = CB_color_cycle[0], ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
            # plt.errorbar(R_test[:,0], get_percentile(cond_preds_1d, 0.5), yerr=err, color = CB_color_cycle[0], ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
        if mf_preds: 
            mf_means = mf_preds.mean[:,0]
            mf_var = mf_preds.covariance_matrix[:,0,0]
            plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, fillstyle = "none", ls = "None", capsize=5, color = CB_color_cycle[1], label = "mean field prediction")
        # plt.scatter(R_test[:,0], y_hat[:,0], marker = "x",  color = CB_color_cycle[2], label = "PLS")        
        plt.legend()
        # plt.savefig(f"examples/eiot/figures/toy_eiot_M={M}_Ntrain={N_train}_Ntest={N_test}_Cobs={C_obs}_Cunobs={C_unobs}__P={P}_sigma={sigma}.pdf")
        plt.show()


def train_lbfgs(model, D_train, R_train,D_test, steps = 3 , plot = False, params=None, convergence_level = None):
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            )
        loss.backward()
        return loss
    loss_list = []
    if params is None:
        params = list(model.parameters())
    for param in params:
        param.data = param.data.contiguous()
    # optimizer = torch.optim.LBFGS([model._mu_u, model._raw_chol])
    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.LBFGS(params=params, line_search_fn="strong_wolfe")
    iterator = tqdm.tqdm(range(steps))
    for _ in iterator:
        loss = -model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        loss_list.append(model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            ).detach().item())
        iterator.set_postfix(loss=loss_list[-1])
    if convergence_level is not None:
         while loss_list[-1] - loss_list[-2] > convergence_level:
            loss = -model.elbo(
                D_train=D_train,
                R_train=R_train, 
                D_test=D_test,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure)
            loss_list.append(model.elbo(
                D_train=D_train,
                R_train=R_train, 
                D_test=D_test,
            ).detach().item())
            print(loss_list[-1])
    if plot:
        with torch.no_grad():
            plt.plot(loss_list)
            plt.show()
    return loss_list


def get_both_init_values_for_latent_variables(D_train, R_train, D_test, num_latent_dimensions):
    D_combined = torch.vstack((D_train,D_test))
    # U, _, _ = torch.linalg.svd(D_combined)
    S_hat = (torch.linalg.inv((R_train.T @ R_train)) @ (R_train.T @ D_train))
    R_hat = D_combined @ torch.linalg.pinv(S_hat)
    E = D_combined - R_hat@S_hat
    # U, _, _ = torch.linalg.svd(D_combined)
    U, _, _ = torch.linalg.svd(E)
    U_scaled = U[:,:num_latent_dimensions]/torch.sqrt(U[:, :num_latent_dimensions].var(axis = 0))
    return U_scaled[:D_train.shape[0]], U_scaled[D_train.shape[0]:]

def get_both_pca_values(D_train, R_train, D_test, num_latent_dimensions):
    D_combined = torch.vstack((D_train,D_test))
    U, _, _ = torch.linalg.svd(D_combined)
    U_scaled = U[:,:num_latent_dimensions]/torch.sqrt(U[:, :num_latent_dimensions].var(axis = 0))
    return U_scaled[:D_train.shape[0]], U_scaled[D_train.shape[0]:]


def train_gd(model, D_train, R_train, D_test, n_steps, plot = False, params = None, lr=None):
    if params is None:
        params = model.parameters()
    if lr:
        optimizer = torch.optim.Adam(params, lr=lr)
    else: 
        optimizer = torch.optim.Adam(params)
    # optimizer = torch.optim.Adam(params, lr=lr)
    # optimizer = torch.optim.Rprop(params, lr=lr)
    ll = []
    # Define a closure function for the optimizer

    # @torch.compile(fullgraph=False)
    # def step():
    #     optimizer.step()

    itterator = tqdm.tqdm(range(n_steps))
    for epoch in itterator:  
        optimizer.zero_grad()
        loss = - model.elbo(D_train=D_train,R_train=R_train, D_test=D_test)
       
        loss.backward()
        optimizer.step()
        # step()
        ll.append(model.elbo( D_train=D_train, R_train=R_train, D_test=D_test).detach().item())
        # if epoch % 10 == 0:
        #     print("cache cleared ")
        #     torch.cuda.empty_cache()
        itterator.set_postfix(loss=-loss.item())
    
    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.show()
    return ll 

def _generate_regular_simplex(n):
    """
    Generate the vertices of a regular simplex in n dimensions centered at the origin.
    Returns an array of shape (n + 1, n).
    """
    # Initialize the vertices matrix
    if n < 1:
        raise ValueError()
    if n == 1:
        return torch.tensor([[1],[-1]])
    else: 
        vertices = _generate_regular_simplex(n-1)
        vertices = torch.hstack([-1/n*torch.ones(n).reshape(-1,1), np.sqrt(1-1/n**2)*vertices])
        new_vector = torch.tensor([1]+[0]*(n-1))
        vertices = torch.vstack([new_vector, vertices])
        return vertices
    
def generate_regular_simplex(n):
    return np.sqrt(n)/(np.sqrt(1+n))*_generate_regular_simplex(n)

def trim_wavelengths(min_wavelengths, max_wavelengths, wavelengths, spectra):
    filter = torch.logical_and(wavelengths<max_wavelengths , min_wavelengths<wavelengths)
    return  spectra[:,filter], wavelengths[filter]


def train(model, training_targets, test_targets, training_inputs , test_inputs, training_weights, steps = 50, lr = 1e-1, frac_of_test=1):
    loss_list = []
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    iterator = tqdm.tqdm(range(steps))
    for _ in iterator:
        loss = -model.elbo(
            training_targets=training_targets,
            training_weights=training_weights, 
            training_inputs=training_inputs,
            test_targets=test_targets[::frac_of_test],
            test_inputs=test_inputs[::frac_of_test], 
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss)
        loss_list.append(loss)
    return loss_list

# def train_lbfgs(model, D_train, D_test, R_train, steps = 3 ):
#     def closure():
#         optimizer.zero_grad()
#         loss = -model.elbo(
#             D_train=D_train,
#             R_train=R_train, 
#             D_test=D_test,
#             )
#         loss.backward()
#         return loss
#     loss_list = []
#     for param in model.parameters():
#         param.data = param.data.contiguous()
#     optimizer = torch.optim.LBFGS(params=model.parameters())
#     iterator = tqdm.tqdm(range(steps))
#     for _ in iterator:
#         loss = -model.elbo(
#             D_train=D_train,
#             R_train=R_train, 
#             D_test=D_test,
#             )
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step(closure)
#         loss_list.append(loss)
#         iterator.set_postfix(loss=loss_list[-1])
#     return loss_list


def get_percentile(dists_1d, alpha):
        def f(a):
            return dists_1d.cdf(a).mean(axis = 0) - alpha
        return fsolve(f,dists_1d.mean().mean(axis=0))

import torch
import tqdm

def trim_wavelengths(min_wavelengths, max_wavelengths, wavelengths, spectra):
    filter = torch.logical_and(wavelengths<max_wavelengths , min_wavelengths<wavelengths)
    return  spectra[:,filter], wavelengths[filter]

def get_init_values_for_latent_variables(D_train, R_train, D_target, num_latent_dimensions):
    S_hat = (torch.linalg.inv((R_train.T @ R_train)) @ (R_train.T @ D_train))
    R_hat = D_target @ torch.linalg.pinv(S_hat)
    E = D_target - R_hat@S_hat
    U, _, _ = torch.linalg.svd(E)
    return U[:,:num_latent_dimensions]/torch.sqrt(U[:, :num_latent_dimensions].var(axis = 0))



def train_obs_inputs(model, training_targets, test_targets, training_inputs , test_inputs, training_weights, steps = 50, lr = 1e-1, frac_of_test=1):
    loss_list = []
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    iterator = tqdm.tqdm(range(steps))
    for _ in iterator:
        loss = -model.elbo(
            training_targets=training_targets,
            training_weights=training_weights, 
            training_inputs=training_inputs,
            test_targets=test_targets[::frac_of_test],
            test_inputs=test_inputs[::frac_of_test], 
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss)
        loss_list.append(loss)
    return loss_list

def train_lbfgs_obs_inputs(model, training_targets, test_targets, training_inputs , test_inputs, training_weights, steps = 3 ):
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo(
            training_targets=training_targets,
            # training_spectra=D_train,
            training_weights=training_weights, 
            training_inputs=training_inputs,
            test_targets=test_targets,
            # test_spectra=D_test[::frac_of_train],
            test_inputs=test_inputs, 
            # wavelengths=wavelengths
            )
        loss.backward()
        return loss
    loss_list = []
    optimizer = torch.optim.LBFGS(params=model.parameters().contiguous())
    iterator = tqdm.tqdm(range(steps))
    for _ in iterator:
        loss = -model.elbo(
            training_targets=training_targets,
            # training_spectra=D_train,
            training_weights=training_weights, 
            training_inputs=training_inputs,
            test_targets=test_targets,
            # test_spectra=D_test[::frac_of_train],
            test_inputs=test_inputs, 
            # wavelengths=wavelengths
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        iterator.set_postfix(loss=loss[-1])
        loss_list.append(loss)
    return loss_list

