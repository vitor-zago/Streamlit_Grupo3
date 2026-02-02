# -*- coding: utf-8 -*-
import streamlit as st
import math
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
st.set_page_config(page_title="DDPM Pro Lab", layout="wide")
st.title("🚀 Diffusion Pro Lab: T=500 + Previsão x0")

# Parâmetros Globais
T = 500 
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ============================
# 2) Inicialização do Session State
# ============================
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "gallery" not in st.session_state:
    st.session_state.gallery = []

# ============================
# 3) Arquitetura da Rede (U-Net Simplificada para Convergência Rápida)
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
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.gn = nn.GroupNorm(4, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.gn(self.conv1(x)))
        h = h + self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = self.act(self.gn(self.conv2(h)))
        return h

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1, stride=2), nn.SiLU())
        self.mid = ResidualBlock(base*2, base*2, time_dim)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base*2, base, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, 64))
        x1 = self.in_conv(x)
        x2 = self.down(x1)
        x3 = self.mid(x2, t_emb)
        x4 = self.up(x3)
        return self.out_conv(x4 + x1) # Skip connection global

if "model" not in st.session_state:
    st.session_state.model = MiniUNet().to(device)

# ============================
# 4) Lógica de Difusão
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 5) Sidebar: Treino & Diagnóstico
# ============================
st.sidebar.header("🛠️ Configuração")
dataset_name = st.sidebar.selectbox("Dataset", ["MNIST", "FashionMNIST"])
epochs = st.sidebar.slider("Épocas", 1, 50, 10)
lr = st.sidebar.select_slider("Taxa de Aprendizado", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)

if st.sidebar.button("🚀 Treinar"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    ds = ds_class(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(Subset(ds, np.random.choice(len(ds), 4000, False)), batch_size=128, shuffle=True)
    
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=lr)
    st.session_state.model.train()
    
    prog = st.progress(0)
    for ep in range(epochs):
        losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        st.session_state.loss_history.append(np.mean(losses))
        prog.progress((ep + 1) / epochs)

if st.session_state.loss_history:
    st.sidebar.subheader("📈 Loss")
    st.sidebar.line_chart(st.session_state.loss_history)

# ============================
# 6) Amostragem com Previsão x0
# ============================
st.header("🖼️ Processo de Geração (xt vs Previsão x0)")
st.write("A linha de baixo mostra o que a IA está tentando 'enxergar' por trás do ruído.")

if st.button("✨ Gerar Nova Imagem"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    
    steps_to_show = [499, 350, 200, 100, 0]
    cols = st.columns(len(steps_to_show))
    col_idx = 0

    for t_inv in range(T-1, -1, -1):
        t_step = torch.full((1,), t_inv, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = st.session_state.model(x, t_step)
            a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
            
            # Previsão x0: Onde a mágica acontece
            pred_x0 = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Próximo passo xt-1
            a = alpha[t_step].view(-1, 1, 1, 1)
            z = torch.randn_like(x) if t_inv > 0 else 0
            x = torch.sqrt(a) * pred_x0 + torch.sqrt(1 - a) * z
            x = torch.clamp(x, -1, 1)

            if t_inv in steps_to_show:
                with cols[col_idx]:
                    st.caption(f"Passo t={t_inv}")
                    st.image((x.squeeze().cpu().numpy()+1)/2, caption="Ruído (xt)", use_container_width=True)
                    st.image((pred_x0.squeeze().cpu().numpy()+1)/2, caption="Previsão (x0)", use_container_width=True)
                col_idx += 1
    
    st.session_state.gallery.append((x.squeeze().cpu().numpy()+1)/2)

# ============================
# 7) Galeria
# ============================
if st.session_state.gallery:
    st.divider()
    st.subheader("🗂️ Galeria")
    g_cols = st.columns(6)
    for i, img in enumerate(reversed(st.session_state.gallery)):
        if i < 12: g_cols[i % 6].image(img, use_container_width=True)
