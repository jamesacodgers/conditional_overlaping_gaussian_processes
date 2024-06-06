import torch
import tqdm

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