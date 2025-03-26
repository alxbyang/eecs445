"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""

import torch
import matplotlib.pyplot as plt

from dataset_challenge import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import config, set_random_seed, make_training_plot


def freeze_layers(model: torch.nn.Module, num_layers: int = 0) -> None:
    for name, param in model.named_parameters():
        if "conv" in name:
            layer = int(name[4])
            if layer <= num_layers:
                param.requires_grad = False


def main():
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("challenge.batch_size"),
    )
    
    # Model
    model = Challenge()    
    model, _, _ = restore_checkpoint(
        model,
        config("source_challenge.checkpoint"),
        force=True,
        pretrain=True,
    )
    freeze_layers(model, 3)

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    axes = make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

        # Updates early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig("challenge_training_plot.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
