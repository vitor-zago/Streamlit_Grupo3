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
st.title("🚀 Diffusion Pro Lab: T=500 + Diagnósticos")

T = 500 
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ============================
# 2) Inicialização do Estado (Session State)
# ============================
if "model" not in st.session_state:
    # Definiremos a classe antes, mas o estado precisa ser inicializado
    pass 

if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

if "gallery" not in st.session_state:
    st.session_state.gallery = []

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

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1, stride=2), nn.SiLU())
        self.bot1 = ResidualBlock(base*2, base*2, time_dim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base*2, base, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.bot1(x, t_emb)
        x = self.up1(x)
        return self.out_conv(x)

if "model" not in st.session_state:
    st.session_state.model = MiniUNet().to(device)

# ============================
# 4) Agendamento de Ruído
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 5) Painel Lateral & Treino (Gráfico de Loss)
# ============================
st.sidebar.header("🛠️ Treinamento")
dataset_name = st.sidebar.selectbox("Dataset", ["MNIST", "FashionMNIST"])
epochs = st.sidebar.slider("Épocas", 1, 30, 5)

if st.sidebar.button("🚀 Iniciar Treino"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    ds = ds_class(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(Subset(ds, np.random.choice(len(ds), 4000, False)), batch_size=128, shuffle=True)
    
    optimizer = torch.optim.AdamW(st.session_state.model.parameters(), lr=1e-3)
    st.session_state.model.train()
    
    for ep in range(epochs):
        epoch_losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_losses.append(loss.item())
        st.session_state.loss_history.append(np.mean(epoch_losses))
    st.sidebar.success("Treino concluído!")

if st.session_state.loss_history:
    st.sidebar.subheader("📈 Curva de Aprendizado")
    st.sidebar.line_chart(st.session_state.loss_history)

# ============================
# 6) Amostragem com Previsão x0
# ============================
st.header("🖼️ Geração e Evolução (Previsão Interna $x_0$)")
st.markdown("A linha superior é a imagem ruidosa atual ($x_t$). A linha inferior é o que a rede **acha** que é a imagem limpa ($x_0$ predicted) naquele momento.")

if st.button("✨ Gerar Sequência de Difusão"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    
    snapshots_xt = []
    snapshots_x0 = []
    steps_to_show = [499, 300, 150, 50, 0]
    
    for t_inv in range(T-1, -1, -1):
        t_step = torch.full((1,), t_inv, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = st.session_state.model(x, t_step)
            a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
            
            # Cálculo da Previsão x0 (A "mágica" interna)
            predicted_x0 = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
            predicted_x0 = torch.clamp(predicted_x0, -1, 1)
            
            # Passo de amostragem padrão
            a = alpha[t_step].view(-1, 1, 1, 1)
            z = torch.randn_like(x) if t_inv > 0 else 0
            x = torch.sqrt(a) * predicted_x0 + torch.sqrt(1 - a) * z
            x = torch.clamp(x, -1, 1)

            if t_inv in steps_to_show:
                snapshots_xt.append((x.clone(), t_inv))
                snapshots_x0.append(predicted_x0.clone())

    # Exibição das Colunas (Processo t -> 0)
    cols = st.columns(len(snapshots_xt))
    for i in range(len(snapshots_xt)):
        with cols[i]:
            st.caption(f"t={snapshots_xt[i][1]}")
            st.image((snapshots_xt[i][0].squeeze().cpu().numpy()+1)/2, caption="Estado xt", use_container_width=True)
            st.image((snapshots_x0[i].squeeze().cpu().numpy()+1)/2, caption="Previsão x0", use_container_width=True)
    
    # Salvar na Galeria
    final_img = (x.squeeze().cpu().numpy()+1)/2
    st.session_state.gallery.append(final_img)

# ============================
# 7) Galeria de Imagens
# ============================
if st.session_state.gallery:
    st.divider()
    st.header("🗂️ Galeria de Gerações Anteriores")
    gal_cols = st.columns(6)
    for idx, img in enumerate(reversed(st.session_state.gallery)):
        if idx < 18: # Limite para não poluir
            gal_cols[idx % 6].image(img, use_container_width=True)
