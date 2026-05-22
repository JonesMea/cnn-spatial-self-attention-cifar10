# DD2424 Project

This project trains a CNN for CIFAR-10 classification using PyTorch. The model is a patchify-style convolutional network with VGG-like blocks and optional spatial self-attention layers.

## Files

- `main.py` contains the experiment configuration and training entry point.
- `models.py` contains `VGGBlock`, `SpatialSelfAttention`, and `ConvNet`.
- `data.py` builds the CIFAR-10 train, validation, train-evaluation, and test loaders.
- `train.py` contains the training loop, evaluation loop, optimizer, scheduler, and history tracking.
- `utils.py` contains seeding, metric helpers, parameter counting, and plotting.
- `cnn_attention_cifar10.py` is a compatibility wrapper that runs `main.py`.

## Requirements

Install the required Python packages:

```powershell
pip install torch torchvision matplotlib
```

If you use CUDA, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Data

By default, the script expects CIFAR-10 to already exist under:

```text
../Datasets
```

This is controlled in `main.py`:

```python
data_dir = "../Datasets"
download = False
```

Set `download = True` if you want torchvision to download CIFAR-10 automatically.

## Running

Run the main experiment with:

```powershell
python .\main.py
```

The original filename still works too:

```powershell
python .\cnn_attention_cifar10.py
```

The script prints training and validation metrics every epoch, evaluates on the test set at the end, and plots loss and accuracy curves.

## Configuration

Most settings are near the top of `main()` in `main.py`, including:

- data settings: `batch_size`, `val_size`, `num_workers`, `augment`
- model settings: `f`, `patch_filters`, `channels`, `conv_dropout`, `dropout`
- attention settings: `use_attention`, `attention_layers`, `attention_heads`, `attention_dropout`
- training settings: `epochs`, `lr`, `weight_decay`, `label_smoothing`

To enable attention, set:

```python
use_attention = True
attention_layers = [0, 1, 2]
attention_heads = 4
```

Each selected attention layer is inserted after the corresponding VGG block.

## Attention Grid Search

`main.py` includes a minimal attention grid search for quick comparison runs. Enable it with:

```python
run_grid_search = True
```

The grid search uses `grid_search_epochs` and compares:

- attention after the last VGG block
- attention after all VGG blocks
- attention after all VGG blocks with attention dropout

The no-attention baseline is not rerun. The script prints the known baseline result for reference:

```text
val loss 0.3839, val acc 0.9040
test loss 0.4334, test acc 0.8989
```

The final test set is not used during the grid search; attention runs are ranked by validation accuracy.
