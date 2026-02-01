# -*- coding: utf-8 -*-
import streamlit as st
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- STREAMLIT HEADER ---
st.set_page_config(page_title="Mini-DDPM MNIST", layout="wide")
st.title("🚀 Modelo de Difusão Mini-DDPM para MNIST")
st.sidebar.header("Configurações")

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
st.sidebar.write(f"**Device:** {device}")

# ============================
# 2) Dataset MNIST
# ============================
@st.cache_resource 
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset_size = 5000
    indices = np.random.choice(len(train_ds), size=subset_size, replace=False)
    return Subset(train_ds, indices)

train_subset = load_data()
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)

# Helper para plotar no Streamlit
def show_images_st(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    axes = axes.flatten()
    for i in range(n):
        img = x[i].squeeze(0)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    st.pyplot(fig)

# ============================
# 3) Noise Schedule
# ============================
T = st.sidebar.slider("Passos de Difusão (T)", 50, 500, 200)
beta_start, beta_end = 1e-4, 0.02
beta = torch.linspace(beta_start, beta_end, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

# ============================
# 4) Processo direto
# ============================
def q_sample(x0, t, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 5) Modelo U-Net (Simplificado para o post)
# ============================
def sinusoidal_time_embedding(t, dim=64):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

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
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = self.act(self.gn2(self.conv2(h)))
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
        x = torch.cat([self.up(x), skip], dim=1)
        return self.block(x, t_emb)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = Down(base, base*2, time_dim)
        self.down2 = Down(base*2, base*4, time_dim)
        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.up2 = Up(base*4 + base*4, base*2, time_dim)
        self.up1 = Up(base*2 + base*2, base, time_dim)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.bot1(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out_conv(x)

# ============================
# 6) Treinamento & Amostragem
# ============================
@torch.no_grad()
def p_sample_loop(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    for t_inv in range(T-1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        a = alpha[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x) if t_inv > 0 else 0
        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z
    return x

# --- INTERFACE PRINCIPAL ---
col1, col2 = st.columns(2)

if st.sidebar.button("Treinar Modelo"):
    model = MiniUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    epochs = st.sidebar.number_input("Épocas", 1, 10, 2)
    
    progress_bar = st.progress(0)
    for ep in range(epochs):
        losses = []
        for x0, _ in train_loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            loss = F.mse_loss(model(xt, t), eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        progress_bar.progress((ep + 1) / epochs)
        st.write(f"Época {ep+1} - Loss: {np.mean(losses):.4f}")

    st.success("Treinamento concluído!")
    
    st.subheader("Gerando novos dígitos...")
    samples = p_sample_loop(model)
    show_images_st(samples, "Dígitos Gerados")

else:
    st.info("Clique no botão à esquerda para iniciar o treinamento.")

