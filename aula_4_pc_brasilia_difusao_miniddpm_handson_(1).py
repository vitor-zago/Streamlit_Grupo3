# -*- coding: utf-8 -*-
"""Aula_4_PC_Brasilia_Difusao_MiniDDPM_HandsOn.py

Modelo de Difusão Mini-DDPM para MNIST - Versão Script Python
"""

import math
import os
import random
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================
# 1) Setup do ambiente
# ============================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

plt.rcParams["figure.dpi"] = 120

# ============================
# 2) Dataset MNIST (imagens 28x28)
# ============================
# Usando o dataset MNIST padrão do PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),                  # [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1] para estabilizar
])

# Carregar MNIST
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Criar subset (para treinamento mais rápido)
subset_size = 10000
indices = np.random.choice(len(train_ds), size=subset_size, replace=False)
train_subset = Subset(train_ds, indices)

# DataLoader
batch_size = 128
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"\n✓ MNIST carregado! {len(train_ds)} imagens no total")
print(f"✓ Usando subset de {len(train_subset)} imagens")

# Função para visualizar imagens
def show_images(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2  # [-1,1] -> [0,1]
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    axes = axes.flatten()
    
    for i in range(n):
        img = x[i].squeeze(0)  # Remove channel dimension for grayscale
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Testar
x0, y0 = next(iter(train_loader))
show_images(x0, "MNIST - Dígitos")
print(f"Batch shape: {x0.shape}")
print(f"Labels: {y0[:10].tolist()}")

# ============================
# 3) Noise Schedule (β, α, ᾱ)
# ============================
T = 200  # passos

beta_start = 1e-4
beta_end = 0.02
beta = torch.linspace(beta_start, beta_end, T)  # (T,)

alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)         # (T,)

beta = beta.to(device)
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

# Visualizar noise schedule
plt.figure()
plt.plot(beta.detach().cpu().numpy())
plt.title("beta_t (noise schedule)")
plt.show()

plt.figure()
plt.plot(alpha_bar.detach().cpu().numpy())
plt.title("alpha_bar_t (sinal preservado acumulado)")
plt.show()

print("beta[0], beta[-1]:", float(beta[0]), float(beta[-1]))
print("alpha_bar[0], alpha_bar[-1]:", float(alpha_bar[0]), float(alpha_bar[-1]))

# ============================
# 4) Processo direto: q(x_t | x_0)
# ============================
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# Demonstrar o processo direto
x0, _ = next(iter(train_loader))
x0 = x0.to(device)

ts = torch.tensor([0, 20, 60, 120, 199], device=device)
imgs = []
for tval in ts:
    t = torch.full((x0.size(0),), int(tval.item()), device=device, dtype=torch.long)
    xt, _ = q_sample(x0, t)
    imgs.append(xt)

plt.figure(figsize=(10, 2))
for i, tval in enumerate(ts):
    plt.subplot(1, len(ts), i+1)
    im = imgs[i][0].detach().cpu()
    im = (im + 1) / 2
    plt.imshow(im.squeeze(0), cmap="gray")
    plt.title(f"t={int(tval)}")
    plt.axis("off")
plt.suptitle("Processo direto: adicionando ruído (x0 -> xt)")
plt.show()

# ============================
# 5.1) Embedding de tempo (sinusoidal)
# ============================
def sinusoidal_time_embedding(t, dim=64):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ============================
# 5.2) U-Net pequena (MNIST)
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
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)

        t = self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = h + t

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, time_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t_emb):
        h = self.block(x, t_emb)
        return self.pool(h), h

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ResidualBlock(in_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.down1 = Down(base, base*2, time_dim)      # 28 -> 14
        self.down2 = Down(base*2, base*4, time_dim)    # 14 -> 7

        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.bot2 = ResidualBlock(base*4, base*4, time_dim)

        self.up2 = Up(base*4 + base*4, base*2, time_dim)  # 7 -> 14
        self.up1 = Up(base*2 + base*2, base, time_dim)    # 14 -> 28

        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)  # prevê epsilon

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, dim=self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)

        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        return self.out_conv(x)

model = MiniUNet(time_dim=64, base=32).to(device)
print("Modelo criado!")

# ============================
# 6) Loop de treinamento
# ============================
lr = 2e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def train_one_epoch(model, loader):
    model.train()
    losses = []
    for x0, _ in loader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        xt, eps = q_sample(x0, t)
        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

# Treinar por algumas épocas
epochs = 2
history = []
for ep in range(1, epochs+1):
    mean_loss = train_one_epoch(model, train_loader)
    history.append(mean_loss)
    print(f"Epoch {ep}/{epochs} - loss: {mean_loss:.4f}")

plt.figure()
plt.plot(history, marker="o")
plt.title("Loss por época (MSE de epsilon)")
plt.xlabel("época")
plt.ylabel("loss")
plt.show()

# Salvar checkpoint
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/mini_ddpm_mnist.pth"
torch.save({
    "model_state": model.state_dict(),
    "T": T,
    "beta_start": beta_start,
    "beta_end": beta_end
}, cpt_path)
print("Checkpoint salvo em:", ckpt_path)

# ============================
# 7) Amostragem (gerar imagens a partir de ruído)
# ============================
@torch.no_grad()
def p_sample_loop(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    frames = []

    for t_inv in range(T-1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps_pred = model(x, t)

        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)

        a = alpha[t].view(-1, 1, 1, 1)
        if t_inv > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z

        if t_inv in [T-1, int(T*0.75), int(T*0.5), int(T*0.25), 0]:
            frames.append(x.clone())

    return x, frames

samples, frames = p_sample_loop(model, n=16)
show_images(samples, "Amostras geradas (difusão reversa)", n=16)

# Mostrar evolução
plt.figure(figsize=(10, 6))
for i, f in enumerate(frames):
    plt.subplot(len(frames), 1, i+1)
    grid = torch.cat([(img.detach().cpu()+1)/2 for img in f[:16]], dim=2)
    plt.imshow(grid.squeeze(0), cmap="gray")
    plt.axis("off")
    plt.title(f"Evolução (frame {i+1}/{len(frames)})")
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("RESPOSTAS ÀS PERGUNTAS:")
print("="*60)
print("1. O que beta_t controla?")
print("   Define a quantidade de ruído adicionada em cada passo da difusão direta.")
print("   Valores maiores de beta significam mais ruído adicionado.")
print()
print("2. O que alpha_bar_t representa?")
print("   É o produto acumulado dos (1 - beta_t). Representa quanto do sinal original")
print("   permanece após t passos de adição de ruído.")
print()
print("3. Por que prever epsilon (ruído)?")
print("   É mais estável numericamente prever o ruído do que prever a imagem original.")
print("   A rede aprende a isolar o que é ruído do que é imagem.")
print()
print("4. Onde entra 'aprender distribuição' (ML/DL)?")
print("   O modelo aprende a distribuição dos dados através do treinamento para reverter")
print("   o processo de difusão, transformando ruído em amostras que seguem a")
print("   distribuição estatística das imagens de treino.")
