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
from io import BytesIO

# ============================
# 1) Configurações de Página
# ============================
st.set_page_config(page_title="DDPM T=500 Completo", layout="wide")
st.title("🎨 Mini-DDPM: Modelo de Difusão Completo (T=500)")

# Hiperparâmetros fixos
T = 500 
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ============================
# 2) Agendamento de Ruído (Schedules)
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 3) Arquitetura da Rede (U-Net)
# ============================
def sinusoidal_time_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: emb = F.pad(emb, (0, 1))
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
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
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
# 4) Gerenciamento de Dados e Estado
# ============================
if "model" not in st.session_state:
    st.session_state.model = MiniUNet().to(device)

@st.cache_resource
def get_loader(dataset_name, size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    if dataset_name == "MNIST":
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    else:
        ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(Subset(ds, np.random.choice(len(ds), size, False)), batch_size=128, shuffle=True)

# ============================
# 5) Interface Lateral (Sidebar)
# ============================
st.sidebar.header("🔧 Painel de Controle")
dataset_choice = st.sidebar.selectbox("Escolha o Dataset", ["MNIST", "FashionMNIST"])
epochs = st.sidebar.slider("Número de Épocas", 1, 20, 5)
train_size = st.sidebar.number_input("Tamanho do Subset", 1000, 10000, 3000)

if st.sidebar.button("🚀 Iniciar Treinamento"):
    loader = get_loader(dataset_choice, train_size)
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=2e-4)
    st.session_state.model.train()
    
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for ep in range(epochs):
        losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        status_text.write(f"Época {ep+1}/{epochs} - Loss Médio: {np.mean(losses):.4f}")
        prog_bar.progress((ep + 1) / epochs)
    st.sidebar.success("Treino Finalizado com Sucesso!")

# Download de pesos
ckpt = BytesIO()
torch.save(st.session_state.model.state_dict(), ckpt)
st.sidebar.download_button("💾 Baixar Modelo (.pth)", ckpt.getvalue(), "modelo_t500.pth")

# ============================
# 6) Amostragem e Visualização
# ============================
st.header("🖼️ Geração Reversa (T=500)")
st.write("Veja o modelo removendo o ruído passo a passo:")

if st.button("✨ Gerar Nova Imagem"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    
    # Snapshots para visualização (5 estágios)
    snapshots = []
    snapshot_steps = [499, 375, 250, 125, 0]
    
    with st.spinner("Realizando amostragem..."):
        for t_inv in range(T-1, -1, -1):
            t_step = torch.full((1,), t_inv, device=device, dtype=torch.long)
            with torch.no_grad():
                eps_pred = st.session_state.model(x, t_step)
                a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
                x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
                a = alpha[t_step].view(-1, 1, 1, 1)
                z = torch.randn_like(x) if t_inv > 0 else 0
                x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z
                x = torch.clamp(x, -1, 1) # Impede imagens brancas
            
            if t_inv in snapshot_steps:
                snapshots.append((x.clone(), t_inv))

    # Exibição em colunas
    cols = st.columns(len(snapshots))
    for i, (img_tensor, step_val) in enumerate(snapshots):
        with cols[i]:
            st.caption(f"Passo t={step_val}")
            # Desnormaliza: [-1, 1] -> [0, 1]
            img_final = (img_tensor.squeeze().cpu().numpy() + 1) / 2
            st.image(img_final, use_container_width=True)
