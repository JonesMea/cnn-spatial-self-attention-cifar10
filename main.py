import torch
import torch.nn as nn

from data import get_loaders
from models import ConvNet
from train import evaluate, fit
from utils import plot_history, set_seed


def make_model(
    f,
    patch_filters,
    channels,
    hidden_dim,
    dropout,
    conv_dropout,
    use_attention,
    attention_layers,
    attention_heads,
    attention_dropout
):
    return ConvNet(
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


def run_attention_grid_search(
    train_loader,
    train_eval_loader,
    val_loader,
    model_settings,
    training_settings,
    device,
    seed
):
    baseline_result = {
        "val_loss": 0.3839,
        "val_acc": 0.9040,
        "test_loss": 0.4334,
        "test_acc": 0.8989
    }

    attention_configs = [
        {
            "name": "attention_last_block",
            "use_attention": True,
            "attention_layers": [2],
            "attention_heads": 4,
            "attention_dropout": 0.0
        },
        {
            "name": "attention_all_blocks",
            "use_attention": True,
            "attention_layers": [1, 2],
            "attention_heads": 4,
            "attention_dropout": 0.0
        },
                {
            "name": "attention_all_blocks",
            "use_attention": True,
            "attention_layers": [0, 1, 2],
            "attention_heads": 4,
            "attention_dropout": 0.0
        },
        {
            "name": "attention_all_blocks_dropout",
            "use_attention": True,
            "attention_layers": [0, 1, 2],
            "attention_heads": 4,
            "attention_dropout": 0.1
        },
    ]

    results = []

    for attention_config in attention_configs:
        print("\nGrid search run:", attention_config["name"])
        print("----------------")

        set_seed(seed)

        model = make_model(
            **model_settings,
            use_attention=attention_config["use_attention"],
            attention_layers=attention_config["attention_layers"],
            attention_heads=attention_config["attention_heads"],
            attention_dropout=attention_config["attention_dropout"]
        )

        history = fit(
            model=model,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            device=device,
            **training_settings
        )

        results.append({
            **attention_config,
            "best_val_acc": max(history["val_acc"]),
            "final_val_acc": history["val_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "training_time": history["training_time"]
        })

    results = sorted(results, key=lambda result: result["best_val_acc"], reverse=True)

    print("\nAttention grid search results")
    print("-----------------------------")
    print(
        "Known baseline_no_attention | "
        f"val loss {baseline_result['val_loss']:.4f} | "
        f"val acc {baseline_result['val_acc']:.4f} | "
        f"test loss {baseline_result['test_loss']:.4f} | "
        f"test acc {baseline_result['test_acc']:.4f}"
    )

    for result in results:
        print(
            f"{result['name']:28s} | "
            f"best val acc {result['best_val_acc']:.4f} | "
            f"final val acc {result['final_val_acc']:.4f} | "
            f"final val loss {result['final_val_loss']:.4f} | "
            f"time {result['training_time']:.1f}s"
        )

    return results


def main():
    # General settings
    data_dir = "../Datasets"
    download = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 26
    run_grid_search = False

    # Data settings
    batch_size = 256
    val_size = 5000
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
    lr = 2e-3
    weight_decay = 1e-3
    label_smoothing = 0.0
    grid_search_epochs = 20

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

    model_settings = {
        "f": f,
        "patch_filters": patch_filters,
        "channels": channels,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "conv_dropout": conv_dropout,
    }

    training_settings = {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
    }

    if run_grid_search:
        grid_training_settings = {
            **training_settings,
            "epochs": grid_search_epochs,
        }

        run_attention_grid_search(
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            model_settings=model_settings,
            training_settings=grid_training_settings,
            device=device,
            seed=seed
        )
        return

    model = make_model(
        **model_settings,
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
        device=device,
        **training_settings
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
