import torch
import torch.nn as nn

from data import get_loaders
from models import ConvNet
from train import evaluate, fit
from utils import plot_history, set_seed


def main():
    # General settings
    data_dir = "../Datasets"
    download = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 26

    # Data settings
    batch_size = 256
    val_size = 1000
    num_workers = 8
    augment = True

    # Model settings
    f = 2
    patch_filters = 64

    # Three VGG-blocks
    channels = [64, 128, 256]

    hidden_dim = 128
    conv_dropout = 0.0
    dropout = 0
    use_attention = False
    attention_layers = [0, 1, 2]  # 0-based
    attention_heads = 4
    attention_dropout = 0.0

    # Training settings
    epochs = 100
    lr = 1e-3
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
        dropout=dropout,
        conv_dropout=conv_dropout,
        use_attention=use_attention,
        attention_layers=attention_layers,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout
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
