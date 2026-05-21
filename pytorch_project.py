import math
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class VGGBlock(nn.Module):
    """Conv 3x3 -> ReLU -> Conv 3x3 -> ReLU -> optional MaxPool."""

    def __init__(self, in_channels, out_channels, use_pool=True, dropout=0.2):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]

        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.block(x)


class ConvNet(nn.Module):
    """
    Patchify ConvNet upgraded with VGG-style blocks.

    Example with f=2 and channels=[64]:
        image -> patchify conv -> VGG block -> flatten -> FC -> logits

    Example with channels=[64, 128, 256]:
        image -> patchify conv -> 3 VGG blocks -> flatten -> FC -> logits
    """

    def __init__(self, f=2, patch_filters=64, channels=None, hidden_dim=128, num_classes=10, dropout=0.2):
        super().__init__()

        if channels is None:
            channels = [64]

        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=patch_filters,
                kernel_size=f,
                stride=f
            ),
            nn.BatchNorm2d(patch_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        blocks = []
        in_channels = patch_filters
        
        for i, out_channels in enumerate(channels):
            # For three blocks, the last block usually skips max-pooling.
            use_pool = i != len(channels) - 1 or len(channels) == 1

            blocks.append(VGGBlock(in_channels, out_channels, use_pool=use_pool, dropout=dropout))
            in_channels = out_channels

        self.features = nn.Sequential(
            self.patchify,
            *blocks
        )

        # Automatically infer flatten size to avoid manual shape calculations.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples
    mean_acc = total_correct / total_examples

    return mean_loss, mean_acc


def fit(model, train_loader, train_eval_loader, val_loader, epochs, lr, weight_decay, label_smoothing, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(5, epochs)

    def lr_factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "training_time": 0}

    start_time = time.perf_counter()

    print("Device:", device)
    print("Parameters:", count_parameters(model))

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | lr {current_lr:.6f} | train loss {train_loss:.4f}, acc {train_acc:.4f} | val loss {val_loss:.4f}, acc {val_acc:.4f}")

    history["training_time"] = time.perf_counter() - start_time

    return history


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(data_dir, batch_size, val_size, augment, download, num_workers, seed):
    train_transforms = []

    if augment:
        train_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        ]

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    if augment:
        train_transforms += [
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_transform = transforms.Compose(train_transforms)

    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download
    )

    val_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=test_transforms,
        download=download
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transforms,
        download=download
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_eval_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=test_transforms,
        download=download
    )

    train_set = Subset(train_full, train_indices)
    train_eval_set = Subset(train_eval_full, train_indices[:5000])
    val_set = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    train_eval_loader = DataLoader(
        train_eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, train_eval_loader, val_loader, test_loader


def accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_metric(train_values, val_values, metric_name):
    epochs = range(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, label="Train")
    plt.plot(epochs, val_values, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(history):
    plot_metric(history["train_loss"], history["val_loss"], "Loss")
    plot_metric(history["train_acc"], history["val_acc"], "Accuracy")


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main():
    # General settings
    data_dir = "../Datasets"
    download = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 26

    # Data settings
    batch_size = 512
    val_size = 1000
    num_workers = 6
    augment = True

    # Model settings
    f = 2
    patch_filters = 64

    # Three VGG-blocks
    channels = [64, 128, 256]

    hidden_dim = 128
    dropout = 0.2

    # Training settings
    epochs = 100
    lr = 2e-3 # = 0.001
    weight_decay = 1e-3
    label_smoothing = 0.0

    set_seed(seed)

    train_loader, train_eval_loader, val_loader, test_loader = get_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_size=val_size,
        augment=augment,
        download=download,
        num_workers=num_workers,
        seed=seed
    )

    model = ConvNet(
        f=f,
        patch_filters=patch_filters,
        channels=channels,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        device=device
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal result")
    print("------------")
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Training time:", round(history["training_time"], 1), "seconds")

    plot_history(history)


if __name__ == "__main__":
    main()
