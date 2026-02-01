# -*- coding: utf-8 -*-

import math
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================
# 1) Setup e Configurações
# ============================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Rodando em: {device}")

# Parâmetros Globais
T = 200  # Passos de difusão
batch_size = 128
epochs = 10 # Aumentado para 10 para ver algum resultado real
lr = 2e-4

# ============================
# 2) Dataset (FashionMNIST)
# ============================
transform = transforms.Compose([
    transforms.ToTensor(),                  # [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # Normaliza para [-1,1]
])

# Baixando FashionMNIST (1 canal, 28x28)
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
subset_size = 10000
indices = np.random.choice(len(train_ds), size=subset_size, replace=False)
train_subset = Subset(train_ds, indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

def show_images(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2  # Reverte [-1,1] para [0,1]
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    axes = axes.flatten()
    for i in range(n):
        img = x[i].squeeze() # Remove o canal de cor para o cmap gray
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ============================
# 3) Noise Schedule
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# 

# ============================
# 4) Arquitetura Mini U-Net
# ============================
def sinusoidal_time_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.gn(self.conv1(x)))
        h = h + self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = self.act(self.gn(self.conv2(h)))
        return h + self.res_conv(x)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim*4), nn.SiLU(), nn.Linear(time_dim*4, time_dim))
        
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = nn.Sequential(nn.AvgPool2d(2)) # 28 -> 14
        self.block1 = ResidualBlock(base, base*2, time_dim)
        self.down2 = nn.Sequential(nn.AvgPool2d(2)) # 14 -> 7
        self.block2 = ResidualBlock(base*2, base*4, time_dim)
        
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest") # 7 -> 14
        self.block3 = ResidualBlock(base*4 + base*2, base*2, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest") # 14 -> 28
        self.block4 = ResidualBlock(base*2 + base, base, time_dim)
        
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        x1 = self.in_conv(x)
        x2 = self.block1(self.down1(x1), t_emb)
        x3 = self.block2(self.down2(x2), t_emb)
        
        x = self.block3(torch.cat([self.up1(x3), x2], dim=1), t_emb)
        x = self.block4(torch.cat([self.up2(x), x1], dim=1), t_emb)
        return self.out_conv(x)

model = MiniUNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ============================
# 5) Treinamento
# ============================
print("Iniciando treinamento...")
history = []
for ep in range(1, epochs + 1):
    model.train()
    epoch_loss = []
    for x0, _ in train_loader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        xt, eps = q_sample(x0, t)
        
        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    
    avg_loss = np.mean(epoch_loss)
    history.append(avg_loss)
    print(f"Época {ep}/{epochs} | Loss: {avg_loss:.4f}")

# ============================
# 6) Amostragem (Reverse Diffusion)
# ============================
@torch.no_grad()
def p_sample_loop(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    
    for t_inv in range(T-1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        
        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        a = alpha[t].view(-1, 1, 1, 1)
        
        # Fórmula de x_t-1
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        
        if t_inv > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta[t].view(-1, 1, 1, 1))
            x = torch.sqrt(a) * x + ( (1-a)/torch.sqrt(1-a_bar) ) * eps_pred + sigma * noise
        else:
            x = x0_hat
            
    return x

print("Gerando amostras...")
samples = p_sample_loop(model, n=16)
show_images(samples, "Imagens Geradas pelo DDPM")

# Salvar
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_ddpm_fashion.pth")
