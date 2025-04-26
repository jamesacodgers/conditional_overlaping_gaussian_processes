import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.mf_methods import MFModels, MF_WS_GPLVM
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs, train_sgd_model


import cProfile

torch.set_default_dtype(torch.float64)

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='./examples/classification/data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# print(train_dataset.targets[0])
# print(train_dataset.data[0].float().mean())
# print(train_dataset.data[0].float().var())
# # print(train_dataset.data[0])

n_train = 25
n_test = 25
batch_size=10

train_images = train_dataset.data[:n_train].float()
train_images = train_images - train_images.mean()
train_images = train_images/train_images.std()
train_images_vec = train_images.flatten(start_dim=-2, end_dim=-1)
train_labels_one_hot = torch.nn.functional.one_hot(train_dataset.targets[:n_train])

# %%
test_images = test_dataset.data[:n_test].float()
test_images = test_images - test_images.mean()
test_images = test_images/test_images.std()
test_images_vec = test_images.flatten(start_dim=-2, end_dim=-1)
test_labels = test_dataset.targets[:n_test]
test_labels_one_hot = torch.nn.functional.one_hot(test_dataset.targets[:n_test])
# %%

A=2

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train=train_images_vec,R_train=train_labels_one_hot, D_test=test_images_vec, num_latent_dimensions=A)

V = torch.randn(6,A)


model =MF_WS_GPLVM(C_obs=10, C_unobs=0, M = train_images_vec.shape[-1],V=V,weight_distribution="categorical", mu_x_test=mu_x_test, mu_x_train=mu_x_train)


# cProfile.run("model.elbo(training_targets=train_images_vec, test_targets=test_images_vec, training_weights=train_labels)", sort="cumtime")

loss = train_sgd(model=model, D_train=train_images_vec, D_test=test_images_vec, R_train=train_dataset.targets[:n_train], n_steps=500, lr=0.1)
loss = train_lbfgs(model=model, D_train=train_images_vec, D_test=test_images_vec, R_train=train_dataset.targets[:n_train], n_steps=5, convergence_level=10)

# plt.matshow(train_images.data[0])
# plt.show()
# %%
with torch.no_grad():
    plt.plot(loss)
    plt.show()
# %%
preds = model.predict_weights(test_images_vec)
print(torch.argmax(preds, dim=-1) == test_labels)
# %%
reconstruction = model.get_s_expectation(model.mu_x_train, model.Sigma_x_train)

with torch.no_grad():
    for i in range(10):
        plt.matshow(reconstruction[0,:,i].reshape(28,28))
        plt.show()

# %%
with torch.no_grad():
    # Plot R_train points
    plt.scatter(
        model.mu_x_train[:, 0],  # X-coordinates
        model.mu_x_train[:, 1],  # Y-coordinates
        cmap="plasma",  # Choose a colormap
        vmin=0,
        vmax=1
    )
    # Plot R_test points
    plt.scatter(
        model.mu_x_test[:, 0],  # X-coordinates
        model.mu_x_test[:, 1],  # Y-coordinates
        marker="x",  # Use a different marker for test points
        cmap="plasma",  # Choose a colormap
        vmin=0,
        vmax=1
    )
    # Add a colorbar to provide a scale for the colors
    plt.colorbar(label="Component 1 fraction")
    # plt.savefig(f"figs/toy_example/mf_latent_space_i={independent_latents}.pdf")
    # Display the plot
    plt.show()
# %%
