import torch
import tqdm
from scipy.optimize import fsolve
import numpy as np

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

def train_cogplvm(model, training_targets, test_targets, training_weights, steps = 50, lr = 1e-1, frac_of_test=1):
    loss_list = []
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    iterator = tqdm.tqdm(range(steps))
    for _ in iterator:
        loss = -model.elbo(
            training_targets=training_targets,
            training_weights=training_weights, 
            test_targets=test_targets[::frac_of_test],
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss)
        loss_list.append(loss)
    return loss_list

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

def train_lbfgs(model, training_targets, test_targets, training_inputs , test_inputs, training_weights, steps = 3 ):
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
    optimizer = torch.optim.LBFGS(params=model.parameters())
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


def get_percentile(dists_1d, alpha):
        def f(a):
            return dists_1d.cdf(a).mean(axis = 0) - alpha
        return fsolve(f,dists_1d.mean().mean(axis=0))