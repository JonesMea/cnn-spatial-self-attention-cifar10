import math
import time

import torch
import torch.nn as nn

from utils import count_parameters


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
