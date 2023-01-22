# import numpy as np
import torch
from torchvision import datasets, transforms, utils

def get_mnist_data(batch_size, device, use_test_subset=True):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=preprocess),
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=preprocess),
        batch_size=batch_size,
        shuffle=True)

    # Create pre-processed training and test sets
    X_train = train_loader.dataset.train_data.to(device).reshape(-1, 784).float() / 255
    y_train = train_loader.dataset.train_labels.to(device)
    X_test = test_loader.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = test_loader.dataset.test_labels.to(device)

    return train_loader, test_loader #(X_test, y_test)

def generate(G, num_samples):
    images = G.generate(num_samples)
    grid = images.reshape(num_samples,1,28,28)
    grid = utils.make_grid(grid, nrow=20, padding=0)
    transforms.ToPILImage()(grid).save('Images/Grid.pdf')