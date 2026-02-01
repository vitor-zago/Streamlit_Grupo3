# -*- coding: utf-8 -*-
"""
Mini-DDPM do zero (MNIST)
Aula didática — Difusão passo a passo
Autor: Fabiano Miranda (adaptado/corrigido)
"""

# ============================
# 1) Imports e setup
# ============================

import math
import random
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------
# Seed
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================
# 2) Dataset MNIST
# ============================

transform = transforms.Compose([
    transforms.ToTensor(),          # [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1]
])

train_ds = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

subset_size = 10_000
indices = np.random.choice(len(train_ds), subset_size, replace=False)
train_subset = Subset(train_ds, indices)

train_loader = DataLoader(
    train_subset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

print(f"MNIST carregado: {len(train_subset)} imagens")

# ============================
# 3) Visualização
# ============================

def show_images(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2  # [-1,1] → [0,1]

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(x[i].squeeze(0), cmap="gray")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

x0, _ = next(iter(train_loader))
show_images(x0, "MNIST - Amostras reais")

# ============================
# 4) Noise schedule
# ============================

T = 200
beta_start = 1e-4
beta_end = 0.02

beta = torch.linspace(beta_start, beta_end, T, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# ============================
# 5) Processo direto q(x_t|x0)
# ============================

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
    return xt, noise

# ============================
# 6) Embedding de tempo
# ============================

def sinusoidal_time_embedding(t, dim=64):
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

# ============================
# 7) Mini U-Net
# ============================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.res(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, time_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t):
        h = self.block(x, t)
        return self.pool(h), h

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ResidualBlock(in_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t)

class MiniUNet(nn.Module):
    def __init__(self, base=32, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.down1 = Down(base, base * 2, time_dim)
        self.down2 = Down(base * 2, base * 4, time_dim)

        self.bot1 = ResidualBlock(base * 4, base * 4, time_dim)
        self.bot2 = ResidualBlock(base * 4, base * 4, time_dim)

        self.up2 = Up(base * 8, base * 2, time_dim)
        self.up1 = Up(base * 4, base, time_dim)

        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)

        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        return self.out_conv(x)

model = MiniUNet().to(device)

# ============================
# 8) Treinamento
# ============================

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def train_epoch():
    model.train()
    losses = []

    for x0, _ in train_loader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        xt, eps = q_sample(x0, t)

        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

for epoch in range(1, 3):
    loss = train_epoch()
    print(f"Epoch {epoch} | Loss: {loss:.4f}")

# ============================
# 9) Amostragem (difusão reversa)
# ============================

@torch.no_grad()
def sample(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)

    for t_inv in reversed(range(T)):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps = model(x, t)

        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)

        a = alpha[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x) if t_inv > 0 else torch.zeros_like(x)

        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z

    return x

samples = sample(model)
show_images(samples, "Amostras geradas (Mini-DDPM)")
