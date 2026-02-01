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

# ============================
# 1) Configurações Iniciais
# ============================
st.set_page_config(page_title="Mini-DDPM MNIST", layout="wide")
st.title("🚀 Mini-DDPM: Modelo de Difusão (MNIST)")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Executando em: **{device.upper()}**")

# ============================
# 2) Definição da Arquitetura
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
        self.down1 = Down(base, base*2, time_dim)
        self.down2 = Down(base*2, base*4, time_dim)
        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.bot2 = ResidualBlock(base*4, base*4, time_dim)
        self.up2 = Up(base*4 + base*4, base*2, time_dim)
        self.up1 = Up(base*2 + base*2, base, time_dim)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

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

# ============================
# 3) Difusão e Dados
# ============================
T = 200
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

@st.cache_resource
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    indices = np.random.choice(len(train_ds), size=5000, replace=False) # Reduzido para Streamlit
    subset = Subset(train_ds, indices)
    return DataLoader(subset, batch_size=128, shuffle=True)

# ============================
# 4) Sidebar e Controle
# ============================
st.sidebar.header("Parâmetros")
epochs = st.sidebar.slider("Épocas", 1, 10, 2)
train_btn = st.sidebar.button("Treinar Modelo")

if "model_state" not in st.session_state:
    st.session_state.model_state = MiniUNet().to(device)

# ============================
# 5) Loop de Treino (Streamlit)
# ============================
if train_btn:
    loader = get_dataloader()
    optimizer = torch.optim.AdamW(st.session_state.model_state.parameters(), lr=2e-4)
    progress_bar = st.progress(0)
    
    for ep in range(epochs):
        losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model_state(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        st.write(f"Época {ep+1} - Loss: {avg_loss:.4f}")
        progress_bar.progress((ep + 1) / epochs)
    st.success("Treino Finalizado!")

# ============================
# 6) Amostragem e Visualização
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

st.header("Gerar Imagens")
if st.button("Gerar Amostras"):
    with st.spinner("Gerando..."):
        samples = p_sample_loop(st.session_state.model_state)
        samples = (samples + 1) / 2
        
        fig, axes = plt.subplots(2, 8, figsize=(12, 3))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].squeeze().cpu().numpy(), cmap="gray")
            ax.axis("off")
        st.pyplot(fig)
