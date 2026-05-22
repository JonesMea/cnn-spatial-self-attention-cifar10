from statistics import mean, variance

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
        "val_loss": 0.4062,
        "val_acc": 0.9016,
        "test_loss": 0.4628,
        "test_acc": 0.8933
    }

    attention_layer_options = [
        [0],
        [1],
        [2],
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2],
    ]
    attention_head_options = [2, 4, 8]
    attention_dropout_options = [0.0, 0.05, 0.1]

    attention_configs = []

    for attention_layers in attention_layer_options:
        layer_name = "".join(str(layer) for layer in attention_layers)

        for attention_heads in attention_head_options:
            for attention_dropout in attention_dropout_options:
                attention_configs.append({
                    "name": f"layers_{layer_name}_heads_{attention_heads}_dropout_{attention_dropout}",
                    "use_attention": True,
                    "attention_layers": attention_layers,
                    "attention_heads": attention_heads,
                    "attention_dropout": attention_dropout
                })

    results = []

    print("Attention grid search runs:", len(attention_configs))

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


def run_repeated_validation_comparison(
    train_loader,
    train_eval_loader,
    val_loader,
    test_loader,
    model_settings,
    training_settings,
    device,
    seed,
    runs=5
):
    configs = [
        {
            "name": "baseline_no_attention",
            "use_attention": False,
            "attention_layers": [],
            "attention_heads": 4,
            "attention_dropout": 0.0
        },
        {
            "name": "layers_12_heads_4_dropout_0.1",
            "use_attention": True,
            "attention_layers": [1, 2],
            "attention_heads": 4,
            "attention_dropout": 0.1
        },
        {
            "name": "layers_02_heads_2_dropout_0.0",
            "use_attention": True,
            "attention_layers": [0, 2],
            "attention_heads": 2,
            "attention_dropout": 0.0
        },
        {
            "name": "layers_1_heads_2_dropout_0.0",
            "use_attention": True,
            "attention_layers": [1],
            "attention_heads": 2,
            "attention_dropout": 0.0
        },
        {
            "name": "layers_012_heads_4_dropout_0.0",
            "use_attention": True,
            "attention_layers": [0, 1, 2],
            "attention_heads": 4,
            "attention_dropout": 0.0
        },
    ]

    results = []

    for config in configs:
        best_val_accs = []
        final_val_accs = []
        test_accs = []

        print("\nRepeated validation config:", config["name"])
        print("---------------------------")

        for run in range(runs):
            run_seed = seed + run
            print(f"\nRun {run + 1}/{runs} | seed {run_seed}")

            set_seed(run_seed)

            model = make_model(
                **model_settings,
                use_attention=config["use_attention"],
                attention_layers=config["attention_layers"],
                attention_heads=config["attention_heads"],
                attention_dropout=config["attention_dropout"]
            )

            history = fit(
                model=model,
                train_loader=train_loader,
                train_eval_loader=train_eval_loader,
                val_loader=val_loader,
                device=device,
                **training_settings
            )

            criterion = nn.CrossEntropyLoss(label_smoothing=training_settings["label_smoothing"])
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            best_val_accs.append(max(history["val_acc"]))
            final_val_accs.append(history["val_acc"][-1])
            test_accs.append(test_acc)

            print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

        result = {
            **config,
            "best_val_acc_mean": mean(best_val_accs),
            "best_val_acc_variance": variance(best_val_accs),
            "final_val_acc_mean": mean(final_val_accs),
            "final_val_acc_variance": variance(final_val_accs),
            "test_acc_mean": mean(test_accs),
            "test_acc_variance": variance(test_accs),
        }
        results.append(result)

    results = sorted(results, key=lambda result: result["final_val_acc_mean"], reverse=True)

    print("\nRepeated validation comparison")
    print("------------------------------")

    for result in results:
        print(
            f"{result['name']:32s} | "
            f"best val acc mean {result['best_val_acc_mean']:.4f}, "
            f"var {result['best_val_acc_variance']:.8f} | "
            f"final val acc mean {result['final_val_acc_mean']:.4f}, "
            f"var {result['final_val_acc_variance']:.8f} | "
            f"test acc mean {result['test_acc_mean']:.4f}, "
            f"var {result['test_acc_variance']:.8f}"
        )

    return results


def main():
    # General settings
    data_dir = "../Datasets"
    download = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 26
    run_grid_search = False
    run_repeated_validation = False

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
    use_attention = True
    attention_layers = [0, 1, 2]  # 0-based
    attention_heads = 4
    attention_dropout = 0.0

    # Training settings
    epochs = 100
    lr = 2e-3
    weight_decay = 1e-3
    label_smoothing = 0.0
    grid_search_epochs = 60
    repeated_validation_epochs = 100
    repeated_validation_runs = 5

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

    if run_repeated_validation:
        repeated_training_settings = {
            **training_settings,
            "epochs": repeated_validation_epochs,
        }

        run_repeated_validation_comparison(
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_settings=model_settings,
            training_settings=repeated_training_settings,
            device=device,
            seed=seed,
            runs=repeated_validation_runs
        )
        return

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
