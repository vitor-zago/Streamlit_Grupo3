# -*- coding: utf-8 -*-
import streamlit as st
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================
# 1) Configurações e Hiperparâmetros
# ============================
st.set_page_config(page_title="DDPM Final Fix", layout="wide")
st.title("🎨 Difusão: Do Ruído à Imagem Real")

T = 500 
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()

# ============================
# 2) Agendamento de Ruído (Schedule)
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 3) Modelo U-Net Reforçado
# ============================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().view(-1, 1) * emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x, t_emb):
        h = F.relu(self.bn(self.conv1(x)))
        h = h + self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = F.relu(self.bn(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(64)
        self.t_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128))
        self.in_c = nn.Conv2d(1, 64, 3, padding=1)
        self.b1 = Block(64, 64, 128)
        self.down = nn.MaxPool2d(2)
        self.b2 = Block(64, 128, 128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.b3 = Block(128+64, 64, 128)
        self.out_c = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t_e = self.t_mlp(self.t_embed(t))
        x1 = self.in_c(x)
        x1 = self.b1(x1, t_e)
        x2 = self.down(x1)
        x2 = self.b2(x2, t_e)
        x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.b3(x, t_e)
        return self.out_c(x)

if "model" not in st.session_state:
    st.session_state.model = UNet().to(device)
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "gallery" not in st.session_state:
    st.session_state.gallery = []

# ============================
# 4) Interface e Treino
# ============================
st.sidebar.header("⚙️ Controlo")
dataset_name = st.sidebar.selectbox("Dataset", ["MNIST", "FashionMNIST"])
epochs = st.sidebar.slider("Épocas", 5, 50, 15)
train_btn = st.sidebar.button("🚀 Treinar Agora")

# MOSTRAR AMOSTRAS REAIS (O QUE FALTA!)
if st.sidebar.button("👀 Mostrar Amostras Reais"):
    st.subheader("📸 Exemplos Reais do Dataset")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=6, shuffle=True)
    reais, _ = next(iter(loader))
    cols = st.columns(6)
    for i in range(6):
        cols[i].image((reais[i].squeeze().numpy()+1)/2, use_container_width=True)

if train_btn:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    ds = ds_class(root="./data", train=True, download=True, transform=transform)
    # Aumentamos o subset para 5000 para melhor aprendizado
    loader = DataLoader(Subset(ds, np.random.choice(len(ds), 5000, False)), batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=1e-3)
    
    st.session_state.model.train()
    bar = st.progress(0)
    for ep in range(epochs):
        losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(st.session_state.model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        st.session_state.loss_history.append(np.mean(losses))
        bar.progress((ep+1)/epochs)

if st.session_state.loss_history:
    st.sidebar.line_chart(st.session_state.loss_history)

# ============================
# 5) Geração (xt vs x0)
# ============================
st.header("🖼️ Geração Reversa")
if st.button("✨ Gerar Imagem"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    snapshots = [499, 300, 150, 50, 0]
    cols = st.columns(len(snapshots))
    c_idx = 0

    for t_inv in range(T-1, -1, -1):
        t_step = torch.full((1,), t_inv, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_p = st.session_state.model(x, t_step)
            a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
            # PREVISÃO X0 (FUNDAMENTAL)
            p_x0 = (x - torch.sqrt(1 - a_bar) * eps_p) / torch.sqrt(a_bar)
            p_x0 = torch.clamp(p_x0, -1, 1)
            
            # ATUALIZAÇÃO XT
            a = alpha[t_step].view(-1, 1, 1, 1)
            z = torch.randn_like(x) if t_inv > 0 else 0
            x = torch.sqrt(a) * p_x0 + torch.sqrt(1 - a) * z
            x = torch.clamp(x, -1, 1)

        if t_inv in snapshots:
            with cols[c_idx]:
                st.caption(f"t={t_inv}")
                st.image((x.squeeze().cpu().numpy()+1)/2, caption="Atual (xt)", use_container_width=True)
                st.image((p_x0.squeeze().cpu().numpy()+1)/2, caption="Previsão (x0)", use_container_width=True)
            c_idx += 1
    st.session_state.gallery.append((x.squeeze().cpu().numpy()+1)/2)

# ============================
# 6) Galeria
# ============================
if st.session_state.gallery:
    st.divider()
    st.subheader("🗂️ Histórico de Gerações")
    g_cols = st.columns(6)
    for i, img in enumerate(reversed(st.session_state.gallery)):
        if i < 12: g_cols[i % 6].image(img, use_container_width=True)
