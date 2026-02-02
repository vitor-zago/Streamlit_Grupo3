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
# 1) Configurações de Página
# ============================
st.set_page_config(page_title="DDPM Fix: Anti-Error", layout="wide")
st.title("🎨 Mini-DDPM: Geração Estável (T=500)")

T = 500 
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()

# ============================
# 2) Parâmetros de Difusão
# ============================
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0).to(device)

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 3) Arquitetura U-Net Estabilizada
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

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(64)
        self.t_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128))
        self.inc = nn.Conv2d(1, 64, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.mid = Block(64, 64, 128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t_e = self.t_mlp(self.t_embed(t))
        x_in = self.inc(x)
        x_down = self.down(x_in)
        x_mid = self.mid(x_down, t_e)
        x_up = self.up(x_mid)
        return self.outc(x_up + x_in)

# Session State
if "model" not in st.session_state:
    st.session_state.model = SimpleUNet().to(device)
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "gallery" not in st.session_state:
    st.session_state.gallery = []

# ============================
# 4) Auxiliar de Visualização (SOLUÇÃO DO ERRO)
# ============================
def convert_to_display(tensor):
    """Garante que os dados estejam em [0.0, 1.0] para o st.image"""
    img = tensor.squeeze().cpu().numpy()
    img = (img + 1.0) / 2.0  # Desnormaliza de [-1, 1] para [0, 1]
    return np.clip(img, 0.0, 1.0) # Força o limite para evitar RuntimeError

# ============================
# 5) Treinamento
# ============================
st.sidebar.header("⚙️ Treino")
dataset_name = st.sidebar.selectbox("Dataset", ["MNIST", "FashionMNIST"])
epochs = st.sidebar.slider("Épocas", 1, 50, 10)

if st.sidebar.button("🚀 Iniciar Treino"):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds_class = datasets.MNIST if dataset_name == "MNIST" else datasets.FashionMNIST
    ds = ds_class(root="./data", train=True, download=True, transform=tr)
    loader = DataLoader(Subset(ds, np.random.choice(len(ds), 4000, False)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(st.session_state.model.parameters(), lr=1e-3)
    
    st.session_state.model.train()
    bar = st.progress(0)
    for ep in range(epochs):
        epoch_losses = []
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, eps = q_sample(x0, t)
            eps_pred = st.session_state.model(xt, t)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())
        st.session_state.loss_history.append(np.mean(epoch_losses))
        bar.progress((ep + 1) / epochs)

if st.session_state.loss_history:
    st.sidebar.line_chart(st.session_state.loss_history)

# ============================
# 6) Geração (Com Clamp Ativo)
# ============================
if st.button("✨ Gerar Nova Imagem"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    steps_to_show = [499, 300, 150, 50, 0]
    cols = st.columns(len(steps_to_show))
    col_idx = 0

    for t_inv in range(T-1, -1, -1):
        t_step = torch.full((1,), t_inv, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_p = st.session_state.model(x, t_step)
            a_bar = alpha_bar[t_step].view(-1, 1, 1, 1)
            
            # Previsão x0
            p_x0 = (x - torch.sqrt(1 - a_bar) * eps_p) / torch.sqrt(a_bar)
            p_x0 = torch.clamp(p_x0, -1, 1)
            
            # Passo Reverso
            if t_inv > 0:
                noise = torch.randn_like(x)
                x = (1 / torch.sqrt(alpha[t_inv])) * (x - ((1 - alpha[t_inv]) / torch.sqrt(1 - a_bar)) * eps_p) + torch.sqrt(beta[t_inv]) * noise
            else:
                x = p_x0
            x = torch.clamp(x, -1, 1)

        if t_inv in steps_to_show:
            with cols[col_idx]:
                st.caption(f"t={t_inv}")
                st.image(convert_to_display(x), caption="xt", use_container_width=True)
                st.image(convert_to_display(p_x0), caption="Pred x0", use_container_width=True)
            col_idx += 1
    
    st.session_state.gallery.append(convert_to_display(x))

# Galeria
if st.session_state.gallery:
    st.divider()
    g_cols = st.columns(6)
    for i, img in enumerate(reversed(st.session_state.gallery)):
        if i < 12: g_cols[i % 6].image(img, use_container_width=True)
