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
    if noise is None: 
        noise = torch.randn_like(x0)
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ============================
# 5) Modelo U-Net
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
            nn.Linear(time_dim * 4, time_dim)
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
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out_conv(x)

# ============================
# 6) Amostragem
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
        z = torch.randn_like(x) if t_inv > 0 else torch.zeros_like(x)
        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z
    
    return x

# ============================
# 7) Interface Principal
# ============================
st.sidebar.subheader("Treinamento")
epochs = st.sidebar.number_input("Épocas", 1, 10, 2)
n_samples = st.sidebar.slider("Número de amostras", 1, 32, 16)

# Inicializar estado da sessão
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

col1, col2 = st.columns(2)

with col1:
    st.subheader("Visualização do Processo de Difusão")
    if st.button("Mostrar Processo de Difusão"):
        x0, _ = next(iter(train_loader))
        x0 = x0[:8].to(device)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        ts = [0, 50, 100, 150, 199]
        for i, tval in enumerate(ts[:4]):
            t = torch.full((x0.size(0),), tval, device=device, dtype=torch.long)
            xt, _ = q_sample(x0, t)
            im = xt[0].detach().cpu().squeeze(0)
            im = (im + 1) / 2
            axes[i].imshow(im, cmap="gray")
            axes[i].set_title(f"t={tval}")
            axes[i].axis('off')
        
        for i, tval in enumerate(ts[1:], start=4):
            t = torch.full((x0.size(0),), tval, device=device, dtype=torch.long)
            xt, _ = q_sample(x0, t)
            im = xt[0].detach().cpu().squeeze(0)
            im = (im + 1) / 2
            axes[i].imshow(im, cmap="gray")
            axes[i].set_title(f"t={tval}")
            axes[i].axis('off')
        
        plt.suptitle("Processo Direto: Adição de Ruído")
        plt.tight_layout()
        st.pyplot(fig)

with col2:
    st.subheader("Treinamento e Geração")
    
    if st.button("Treinar Modelo"):
        with st.spinner("Treinando modelo..."):
            model = MiniUNet().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            
            progress_bar = st.progress(0)
            loss_history = []
            
            for ep in range(epochs):
                losses = []
                for batch_idx, (x0, _) in enumerate(train_loader):
                    x0 = x0.to(device)
                    t = torch.randint(0, T, (x0.size(0),), device=device)
                    xt, eps = q_sample(x0, t)
                    eps_pred = model(xt, t)
                    loss = F.mse_loss(eps_pred, eps)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                avg_loss = np.mean(losses)
                loss_history.append(avg_loss)
                progress_bar.progress((ep + 1) / epochs)
                st.write(f"Época {ep+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Salvar modelo na sessão
            st.session_state.model = model
            st.session_state.trained = True
            
            # Plotar histórico de loss
            fig_loss, ax = plt.subplots(figsize=(8, 4))
            ax.plot(loss_history, marker='o')
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss (MSE)")
            ax.set_title("Histórico de Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
            
            st.success("Treinamento concluído!")

# Geração de amostras
if st.session_state.trained and st.session_state.model is not None:
    st.subheader("Gerar Novos Dígitos")
    
    if st.button("Gerar Amostras"):
        with st.spinner("Gerando dígitos a partir do ruído..."):
            samples = p_sample_loop(st.session_state.model, n=n_samples)
            show_images_st(samples, f"Dígitos Gerados (n={n_samples})", n=n_samples)
            
            # Salvar checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = "checkpoints/mini_ddpm_mnist.pth"
            torch.save({
                "model_state": st.session_state.model.state_dict(),
                "T": T,
                "beta_start": beta_start,
                "beta_end": beta_end
            }, ckpt_path)
            st.sidebar.success(f"Checkpoint salvo em: {ckpt_path}")
else:
    st.info("Treine o modelo primeiro para gerar dígitos!")

