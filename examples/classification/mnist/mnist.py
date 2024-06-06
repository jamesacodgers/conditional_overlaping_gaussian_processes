import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.cogplvm import COGPLMV

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

n_train = 100
n_test = 100

train_images = train_dataset.data[:n_train].float()
train_images = train_images - train_images.mean()
train_images = train_images/train_images.std()
train_images_vec = train_images.flatten(start_dim=-2, end_dim=-1)
train_labels = torch.nn.functional.one_hot(train_dataset.targets[:n_train])


test_images = test_dataset.data[:n_test].float()
test_images = test_images - test_images.mean()
test_images = test_images/test_images.std()
test_images_vec = test_images.flatten(start_dim=-2, end_dim=-1)
test_labels = torch.nn.functional.one_hot(test_dataset.targets[:n_test])

model = COGPLMV(C_obs=10, C_unobs=0, num_latent_dims=3, V = torch.randn(17,3), N_train=n_train, N_test=n_test, M = train_images_vec.shape[-1])

model.elbo(training_targets=train_images, test_targets=test_images, training_weights=train_labels)

# plt.matshow(train_images.data[0])
# plt.show()