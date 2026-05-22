import random

import matplotlib.pyplot as plt
import torch


def set_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
