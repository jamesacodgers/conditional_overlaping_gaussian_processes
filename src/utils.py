import torch
import tqdm
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.prob_utils import get_struc_ci

def snv_correction(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)

    # Apply SNV correction
    snv_corrected = (spectra - mean) / std

    return snv_corrected


def train_conditional_model_lbfgs(model, D_train, R_train,  n_steps, D_test=None, plot = False, params = None, lr=None):
    def closure():
        optimizer.zero_grad()
        _optimize_lvs(model=model,D_train=D_train,R_train=R_train, D_test=D_test)
        optimizer.zero_grad()
        loss = -model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            )
        loss.backward()
        return loss
    if params is None:
        params = model.parameters()
    if lr:
        optimizer = torch.optim.Adam(params, lr=lr)
    else: 
        optimizer = torch.optim.Adam(params)
    # optimizer = torch.optim.Adam(params, lr=lr)
    # optimizer = torch.optim.Rprop(params, lr=lr)
    ll = []

    iterator = tqdm.tqdm(range(n_steps))
    for _ in iterator:
        loss = -model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        ll.append(model.elbo(
            D_train=D_train,
            R_train=R_train, 
            D_test=D_test,
            ).detach().item())
        iterator.set_postfix(loss=ll[-1])
    
    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.show()
    return ll 

def _optimize_lvs(model, D_train, R_train, D_test=None):
    # model.mu_x_test = torch.nn.Parameter(torch.zeros_like(model.mu_x_test))
    # model.mu_x_train = torch.nn.Parameter(torch.zeros_like(model.mu_x_train))
    model.mu_x_test = torch.nn.Parameter(torch.randn_like(model.mu_x_test))
    model.mu_x_train = torch.nn.Parameter(torch.randn_like(model.mu_x_train))
    model._log_Sigma_x_test = torch.nn.Parameter(torch.zeros_like(model.mu_x_test))
    model._log_Sigma_x_train = torch.nn.Parameter(torch.zeros_like(model.mu_x_train))
    train_lbfgs(model, D_train=D_train, R_train=R_train, D_test=D_test, n_steps=2, convergence_level=2, params=[
        model.mu_x_test, 
        model.mu_x_train, 
        model._log_Sigma_x_test, 
        model._log_Sigma_x_train
    ])

def train_sgd_model(model, D_train, R_train,  n_steps, D_test=None, plot = False, params = None, lr=None):
    if params is None:
        params = model.parameters()
    if lr:
        optimizer = torch.optim.Adam(params, lr=lr)
    else: 
        optimizer = torch.optim.Adam(params)
    # optimizer = torch.optim.Adam(params, lr=lr)
    # optimizer = torch.optim.Rprop(params, lr=lr)
    ll = []
    itterator = tqdm.tqdm(range(n_steps))
    for epoch in itterator:  
        
        optimizer.zero_grad()
        _optimize_lvs(model=model,D_train=D_train,R_train=R_train, D_test=D_test)
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
    


def plot_preds(R_test, cond_preds: torch.distributions.Distribution = None, mf_preds = None, title=None, save_path = None) -> None: 
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
            plt.scatter(R_test[:,0], means.mean(dim=0))
            # for mean in means:
            #     plt.scatter(R_test[:,0], mean, c="green", alpha=0.1)
            lb,ub = get_struc_ci(cond_preds)
            tot_mean = means.mean(dim=0).numpy()
            err = np.array(np.vstack([tot_mean - lb,ub-tot_mean])) # technically this could fail if mean isn't inside 95% ci
            plt.errorbar(R_test[:,0], means.mean(dim=0), yerr=err, color = CB_color_cycle[0], ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
        if mf_preds: 
            mf_means = mf_preds.mean[:,0]
            mf_var = mf_preds.covariance_matrix[:,0,0]
            plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, fillstyle = "none", ls = "None", capsize=5, color = CB_color_cycle[1], label = "mean field prediction")
        # plt.scatter(R_test[:,0], y_hat[:,0], marker = "x",  color = CB_color_cycle[2], label = "PLS")        
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


def train_lbfgs(model, D_train, R_train,D_test=None, n_steps = 3 , plot = False, params=None, convergence_level = None):
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
    optimizer = torch.optim.LBFGS(params=params, line_search_fn="strong_wolfe")
    iterator = tqdm.tqdm(range(n_steps))
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
    R_train_float = R_train.type(torch.float64)
    D_combined = torch.vstack((D_train,D_test))
    # U, _, _ = torch.linalg.svd(D_combined)
    S_hat = (torch.linalg.inv((R_train_float.T @ R_train_float)) @ (R_train_float.T @ D_train))
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


def train_gd(model, D_train, R_train,  n_steps, D_test=None, plot = False, params = None, lr=None):
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

