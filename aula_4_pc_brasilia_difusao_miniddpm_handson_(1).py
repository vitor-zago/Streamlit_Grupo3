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
# 1) Configurações de Página
# ============================
st.set_page_config(page_title="Mini-DDPM Brasília", layout="wide")
st.title("🧠 Hands-On: Modelos de Difusão (Mini-DDPM)")
st.markdown("Objetivo: Gerar imagens de dígitos/roupas a partir de puro ruído.")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Dispositivo detectado: {device.upper()}")

# ============================
# 2) Arquitetura da Rede (U-Net)
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
# 3) Lógica de Difusão
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

# ============================
# 4) Interface Streamlit
# ============================
if "model" not in st.session_state:
    st.session_state.model = MiniUNet().to(device)

st.sidebar.header("Configuração")
dataset_type = st.sidebar.selectbox("Escolha o Dataset", ["MNIST (Números)", "FashionMNIST (Roupas)"])
n_epochs = st.sidebar.slider("Épocas de Treino", 1, 10, 2)
subset_size = st.sidebar.number_input("Tamanho do Subset (Imagens)", 1000, 10000, 3000)

if st.sidebar.button("🚀 Iniciar Treinamento"):
    # Carregamento de dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1) # Normaliza para [-1, 1]
    ])
    
    with st.spinner(f"Baixando e preparando {dataset_type}..."):
        if dataset_type == "MNIST (Números)":
            full_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        else:
            full_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
            
        indices = np.random.choice(len(full_ds), size=subset_size, replace=False)
        train_loader = DataLoader(Subset(full_ds, indices), batch_size=128, shuffle=True)

    # Loop de Treino
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=2e-4)
    loss_area = st.empty()
    prog_bar = st.progress(0)
    
    st.session_state.model.train()
    for ep in range(n_epochs):
        epoch_losses = []
        for x0, _ in train_loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_l = np.mean(epoch_losses)
        loss_area.write(f"📊 Época {ep+1}/{n_epochs} | Loss: {avg_l:.4f}")
        prog_bar.progress((ep + 1) / n_epochs)
    
    st.success("✅ Treinamento concluído!")

# ============================
# 5) Amostragem (Geração)
# ============================
st.divider()
st.header("🖼️ Geração via Difusão Reversa")
st.write("Aqui o modelo tenta 'limpar' um ruído aleatório para formar uma imagem.")

if st.button("✨ Gerar Novas Imagens"):
    st.session_state.model.eval()
    n_imgs = 16
    # Começa com puro ruído gaussiano
    x = torch.randn(n_imgs, 1, 28, 28, device=device)
    
    # Barra de progresso para a geração (sampling)
    sample_bar = st.progress(0)
    
    for t_inv in range(T-1, -1, -1):
        t_step = torch.full((n_imgs,), t_inv, device=device, dtype=torch.long)
        
        with torch.no_grad():
            eps_pred = st.session_state.model(x, t_step)
            a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
            x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
            a = alpha[t_step].view(-1, 1, 1, 1)
            
            z = torch.randn_like(x) if t_inv > 0 else 0
            x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z
            
            # CORREÇÃO CRÍTICA: Clamp para evitar valores que estouram o branco
            x = torch.clamp(x, -1, 1)
        
        if t_inv % 20 == 0:
            sample_bar.progress((T - t_inv) / T)

    sample_bar.progress(100)
    
    # Exibição final
    # Desnormaliza: [-1, 1] -> [0, 1]
    imgs_to_show = (x + 1) / 2
    imgs_to_show = imgs_to_show.cpu().numpy()
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs_to_show[i].squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
