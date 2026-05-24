# Spatial Multi-Head Self-Attention in VGG-Style ConvNets for CIFAR-10

This repository contains the code and report for a DD2424 Deep Learning project on CIFAR-10 image classification. The project builds a strong VGG-style convolutional baseline and then investigates whether spatial multi-head self-attention modules improve generalization.

## Project summary

The main research question is:

> Does adding spatial multi-head self-attention to a strong VGG-style ConvNet improve CIFAR-10 generalization?

The final baseline is a VGG-style ConvNet trained from scratch using data augmentation, AdamW weight decay, warmup followed by cosine learning-rate decay, and global average pooling. The extension adds spatial self-attention modules after selected VGG blocks and evaluates different attention placements, head counts, and attention dropout values.

The best attention model used attention after all three VGG blocks with four heads and no attention dropout.

## Main results

| Model                 | Setting                   | Test accuracy |
| No-attention baseline | 5-seed mean, 45k/5k split | 89.26%        |
| Best attention model  | 5-seed mean, 45k/5k split | 89.65%        |
| No-attention baseline | Final 49k/1k single run   | 89.74%        |
| Best attention model  | Final 49k/1k single run   | 90.02%        |

The improvement from attention was positive but modest. The results suggest that spatial self-attention can provide a small generalization benefit on top of a strong convolutional feature extractor, but the effect depends on the attention configuration.

## Repository structure

```text
.
├── data.py                  # CIFAR-10 loading, augmentation, and stratified splits
├── models.py                # VGG-style ConvNet and spatial self-attention module
├── train.py                 # Training loop, evaluation, optimizer, scheduler
├── utils.py                 # Seeding, parameter counting, plotting utilities
├── main.py                  # Experiment settings and entry point
├── plots_basic/             # Baseline model learning curves
├── plots_attention/         # Attention-model learning curves
└── README.md
```
