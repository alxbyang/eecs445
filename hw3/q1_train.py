import os, time, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib import pyplot as plt

from q1_model import RNN


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor([self.y[idx]]).float()
    def __len__(self):
        return len(self.X)


def get_train_val_test(batch_size=64):
    f = np.load('q1_data/data.npz')
    X, y = f['X'], f['y']
    print(X.shape, y.shape)
    
    print('Creating splits')
    Xtr, X__, ytr, y__ = train_test_split(X,   y,   train_size=0.8, stratify=y,   random_state=0)
    Xva, Xte, yva, yte = train_test_split(X__, y__, test_size=0.5, stratify=y__, random_state=0)
    
    tr = SimpleDataset(Xtr, ytr)
    va = SimpleDataset(Xva, yva)
    te = SimpleDataset(Xte, yte)
    
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size)
    te_loader = DataLoader(te, batch_size=batch_size)
    
    print('Feature shape, Label shape, Class balance:')
    print('\t', tr_loader.dataset.X.shape, tr_loader.dataset.y.shape, tr_loader.dataset.y.mean())
    print('\t', va_loader.dataset.X.shape, va_loader.dataset.y.shape, va_loader.dataset.y.mean())
    print('\t', te_loader.dataset.X.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    return tr_loader, va_loader, te_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model.train()
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _evaluate_epoch(tr_loader, va_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        # Evaluate on train
        y_true, y_score = [], []
        running_loss = []
        for X, y in tr_loader:
                output = model(X)
                y_true.append(y.numpy())
                y_score.append(output)
                running_loss.append(criterion(output, y).item())

        y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
        train_loss = np.mean(running_loss)
        train_score = metrics.roc_auc_score(y_true, y_score)
        print('train loss', train_loss, 'train AUROC', train_score)

        # Evaluate on validation
        y_true, y_score = [], []
        running_loss = []
        for X, y in va_loader:
            with torch.no_grad():
                output = model(X)
                y_true.append(y.numpy())
                y_score.append(output)
                running_loss.append(criterion(output, y).item())

        y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
        val_loss = np.mean(running_loss)
        val_score = metrics.roc_auc_score(y_true, y_score)
        print('val loss', val_loss, 'val AUROC', val_score)
    return train_loss, val_loss, train_score, val_score


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)


if __name__ == '__main__':
    tr_loader, va_loader, te_loader = get_train_val_test(batch_size=64)
    
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    n_epochs = 30
    learning_rate = 8*1e-4
    
    model = RNN(70, 64, 1)
    print('Number of float-valued parameters:', count_parameters(model))

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    outputs = []

    print('Epoch', 0)
    out = _evaluate_epoch(tr_loader, va_loader, model, criterion)
    outputs.append(out)

    for epoch in range(0, n_epochs):
        print('Epoch', epoch+1)
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        out = _evaluate_epoch(tr_loader, va_loader, model, criterion)
        outputs.append(out)
        
        # Save model parameters
        save_checkpoint(model, epoch+1, 'q1_checkpoint/')

    train_losses, val_losses, train_scores, val_scores = zip(*outputs)
    
    fig, ax = plt.subplots(figsize=(5,5))
    plt.plot(range(n_epochs + 1), train_scores, '--o', label='Train')
    plt.plot(range(n_epochs + 1), val_scores, '--o', label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.savefig('q1_auroc.png', dpi=300)
    
    fig, ax = plt.subplots(figsize=(5,5))
    plt.plot(range(n_epochs + 1), train_losses, '--o', label='Train')
    plt.plot(range(n_epochs + 1), val_losses, '--o', label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('Loss (binary cross entropy)')
    plt.legend()
    plt.savefig('q1_loss.png', dpi=300)
