import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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
