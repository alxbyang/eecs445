"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - HW4 - q4_train.py
Script for training an autoencoder for reconstruction and denoising tasks
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from q4_model import Autoencoder

torch.manual_seed(445)

def retrain(path: str) -> bool:
    """
    Check if a model exists and if the user wants to retrain the model
    
    Args:
    - path: path to the model

    Returns:
    - whether to train from scratch
    """
    train_from_scratch = True
    if os.path.exists(path):
        load_model = None
        while load_model not in ["y", "n"]:
            load_model = input(f"Found a saved model in {path}. Do you want to use this model? (y/n) ")
            load_model = load_model.lower().replace(" ", "")
        train_from_scratch = load_model == "n"
    return train_from_scratch

def plot_pictures(
    x_test: torch.Tensor, 
    decoded_imgs: torch.Tensor, 
    latents: torch.Tensor, 
    filename: str, 
    denoise: bool=False) -> None:
    """
    Plot original, reconstructed, and latent images

    Args:
    - x_test: original images
    - decoded_imgs: reconstructed images
    - latents: latent vectors
    - filename: filename for saving the plot
    - denoise: whether the images are denoised
    """
    plt.figure(figsize=(20, 6))
    n = x_test.shape[0]
    
    for i in range(n):
        # display original + noise
        ax = plt.subplot(3, n, i + 1)
        plt.title("original + noise") if denoise else plt.title("original")
        plt.imshow(x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(3, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(decoded_imgs[i])
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)

        # display latent vectors as images
        cx = plt.subplot(3, n, i + 2*n + 1)
        plt.title("latents")
        plt.imshow(latents[i].reshape(8, 8))
        plt.gray()
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)

    fig = plt.gcf()
    plt.show()
    fig.savefig(f"{filename}.png", format='png')
    plt.close()


def train_autoencoder(
    autoencoder: nn.Module, 
    trainloader: DataLoader, 
    optimizer: optim.Optimizer, 
    denoise: bool=False, 
    epochs: int=10) -> nn.Module:
    """
    Standard pytorch training loop for an autoencoder

    Args:
    - autoencoder: autoencoder model
    - trainloader: DataLoader for training data
    - optimizer: optimizer for training
    
    Returns:
    - trained autoencoder
    """
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        tbar = tqdm(trainloader)
        for images, _ in tbar:
            if images.dim() == 4: images = images.squeeze(1)
            x = images
            
            if denoise: x = add_noise(x)

            optimizer.zero_grad()
            outputs = autoencoder(x)
            if outputs.dim() == 4: outputs = outputs.squeeze(1)
            loss = criterion(outputs, images)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            tbar.set_description(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(trainloader)}')
    
    return autoencoder

def add_noise(images: torch.Tensor, noise_level: float=0.2) -> torch.Tensor:
    """
    Adds noise to input images, scaled by noise_level

    Args:
    - images: input images
    - noise_level: scaling factor for noise

    Returns:
    - images with added noise
    """
    # Hint: consider using the pytorch's `randn_like` method
    # https://pytorch.org/docs/stable/generated/torch.randn_like.html

    # TODO: Add noise to the images
    noise = noise_level * torch.randn_like(images)
    images_modified = images + noise
    images_modified = torch.clamp(images_modified, -1.0, 1.0)
    return images_modified

def test_autoencoder(
    autoencoder: nn.Module, 
    testloader: DataLoader, 
    filename: str, 
    denoise: bool=False) -> None:
    """
    Test the autoencoder on a batch of images and plot the results
    Args:
    - autoencoder: trained autoencoder model
    - testloader: DataLoader for test data
    - filename: filename for saving the plot
    - denoise: whether the images are denoised
    """
    x_test = next(iter(testloader))[0]
    if denoise: x_test = add_noise(x_test)

    decoded_imgs = autoencoder(x_test).detach().cpu().numpy().squeeze()
    latents = autoencoder.encode(x_test).detach().cpu().numpy().squeeze()
    x_test = x_test.detach().cpu().numpy().squeeze()
    
    plot_pictures(x_test, decoded_imgs, latents, filename, denoise)

def part_a(trainloader, testloader):
    """
    Train the autoencoder for part a
    """
    autoencoder = Autoencoder((28, 28), 64)
    if retrain(os.path.join("checkpoints", "autoencoder.pth")):
        print("Starting training for autoencoder (part a), sit tight!")
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        autoencoder = train_autoencoder(autoencoder, trainloader, optimizer, denoise=False, epochs=10)
        torch.save(autoencoder.state_dict(), os.path.join("checkpoints", "autoencoder.pth"))
    else:
        autoencoder.load_state_dict(torch.load(os.path.join("checkpoints", "autoencoder.pth"), weights_only=True))
    test_autoencoder(autoencoder, testloader, 'autoencoder', denoise=False)


def part_b(trainloader, testloader):
    """
    Train the autoencoder with regularization for part b
    """
    autoencoder_reg = Autoencoder((28, 28), 64)
    if retrain(os.path.join("checkpoints", "autoencoder_reg.pth")):
        print("Starting training for regularized autoencoder (part b), sit tight!")
        optimizer = optim.Adam(autoencoder_reg.parameters(), lr=0.001, weight_decay=1e-3)
        autoencoder_reg = train_autoencoder(autoencoder_reg, trainloader, optimizer, denoise=False, epochs=10)
        torch.save(autoencoder_reg.state_dict(), os.path.join("checkpoints", "autoencoder_reg.pth"))
    else:
        autoencoder_reg.load_state_dict(torch.load(os.path.join("checkpoints", "autoencoder_reg.pth"), weights_only=True))
    test_autoencoder(autoencoder_reg, testloader, 'regularized_autoencoder', denoise=False)

def part_c(trainloader, testloader):
    """
    Train the denoising autoencoder for part c
    """
    denoise = Autoencoder((28, 28), 64)
    if retrain(os.path.join("checkpoints", "denoise.pth")):
        print("Starting training for denoise autoencoder (part c), sit tight!")
        optimizer = optim.Adam(denoise.parameters(), lr=0.001)
        denoise = train_autoencoder(denoise, trainloader, optimizer, denoise=True, epochs=10)
        torch.save(denoise.state_dict(), os.path.join("checkpoints", "denoise.pth"))
    else:
        denoise.load_state_dict(torch.load(os.path.join("checkpoints", "denoise.pth"), weights_only=True))
    test_autoencoder(denoise, testloader, 'denoise', denoise=True)

def main():
    os.makedirs("checkpoints", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.FashionMNIST('data', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST('data', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=5, shuffle=False)

    part = False
    while not part:
        part = input("Enter the part to train (a/b/c): ").strip().lower()
        if part == 'a': part_a(trainloader, testloader)
        elif part == 'b': part_b(trainloader, testloader)
        elif part == 'c': part_c(trainloader, testloader)
        else:
            print("Invalid part. Please enter 'a', 'b', or 'c'.")
            part = False

if __name__ == "__main__":
    main()