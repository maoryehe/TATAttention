# src/model/TotalCountModel.py
import torch
import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, in_channels=5, out_channels=64, kernel_size=7, dropout=0.1, same_length=True):
        super().__init__()
        padding = 'same' if same_length else kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.conv(x)   # (B, Cin, L) -> (B, C, L)
        x = self.bn(x); x = self.act(x); x = self.drop(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=5, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout1d(dropout)
        self.short = nn.Identity() if (in_channels == out_channels and stride == 1) \
                     else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        r = self.short(x)
        x = self.conv1(x); x = self.bn1(x); x = self.act(x); x = self.drop(x)
        x = self.conv2(x); x = self.bn2(x)
        return self.act(x + r)

class TotalCountModel(nn.Module):
    """Stem -> Residual -> MultiheadAttention -> MeanPool -> Linear(1)"""
    def __init__(self, cin=5, channels=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.stem = Stem(in_channels=cin, out_channels=channels, kernel_size=7, dropout=dropout, same_length=True)
        self.res  = Residual(in_channels=channels, out_channels=channels, kernel_size=5, stride=1, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.head = nn.Linear(channels, 1)
    def forward(self, x):
        # x: (B, Cin, L)
        x = self.stem(x)          # (B, C, L)
        x = self.res(x)           # (B, C, L)
        x = x.transpose(1, 2)     # (B, L, C)
        y, _ = self.attn(x, x, x) # (B, L, C) — no masks, no PE
        z = y.mean(dim=1)         # (B, C)
        return self.head(z)       # (B, 1)
