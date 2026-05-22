import torch.nn as nn


class VGGBlock(nn.Module):
    """Conv 3x3 -> ReLU -> Conv 3x3 -> ReLU -> optional MaxPool."""

    def __init__(self, in_channels, out_channels, use_pool=True, conv_dropout=0.2):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(conv_dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(conv_dropout)
        ]

        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions in a CNN feature map."""

    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for MultiheadAttention.")

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Treat every spatial position as one token with C features.
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Pre-norm transformer-style residual attention.
        norm_tokens = self.norm(tokens)
        attention_tokens, _ = self.attention(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + attention_tokens

        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class ConvNet(nn.Module):
    """
    Patchify ConvNet upgraded with VGG-style blocks.

    Example with f=2 and channels=[64]:
        image -> patchify conv -> VGG block -> flatten -> FC -> logits

    Example with channels=[64, 128, 256]:
        image -> patchify conv -> 3 VGG blocks -> flatten -> FC -> logits
    """

    def __init__(
        self,
        f=2,
        patch_filters=64,
        channels=None,
        hidden_dim=128,
        num_classes=10,
        conv_dropout=0.2,
        dropout=0.2,
        use_attention=False,
        attention_layers=None,
        attention_heads=4,
        attention_dropout=0.0
    ):
        super().__init__()

        if channels is None:
            channels = [64]

        if attention_layers is None:
            attention_layers = [len(channels) - 1] if use_attention else []
        elif not use_attention:
            attention_layers = []

        attention_layers = set(attention_layers)

        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=patch_filters,
                kernel_size=f,
                stride=f
            ),
            nn.BatchNorm2d(patch_filters),
            nn.ReLU(),
            nn.Dropout2d(conv_dropout)
        )

        blocks = []
        in_channels = patch_filters

        for i, out_channels in enumerate(channels):
            # For three blocks, the last block usually skips max-pooling.
            use_pool = i != len(channels) - 1 or len(channels) == 1

            blocks.append(VGGBlock(in_channels, out_channels, use_pool=use_pool, conv_dropout=conv_dropout))

            if i in attention_layers:
                blocks.append(SpatialSelfAttention(out_channels, num_heads=attention_heads, dropout=attention_dropout))

            in_channels = out_channels

        self.features = nn.Sequential(
            self.patchify,
            *blocks
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
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
