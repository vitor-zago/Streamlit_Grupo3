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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = "cpu"  # Forçar CPU no Streamlit Cloud (mais confiável)
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
    subset_size = 2000  # Reduzido para treinamento mais rápido
    indices = np.random.choice(len(train_ds), size=subset_size, replace=False)
    return Subset(train_ds, indices)

train_subset = load_data()
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)  # Batch menor

# Helper para plotar no Streamlit
def show_images_st(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.5))
    axes = axes.flatten()
    for i in range(min(n, len(axes))):
        img = x[i].squeeze(0)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    st.pyplot(fig, use_container_width=True)

# ============================
# 3) Noise Schedule
# ============================
T = st.sidebar.slider("Passos de Difusão (T)", 50, 200, 100)  # Reduzido
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
# 5) Modelo U-Net (Simplificado)
# ============================
def sinusoidal_time_embedding(t, dim=32):  # Dim reduzida
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
        self.act = nn.ReLU()  # Mais simples que SiLU
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = self.act(self.conv2(h))
        return h + self.res_conv(x)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=32, base=16):  # Base menor
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), 
            nn.ReLU(), 
            nn.Linear(time_dim * 2, time_dim)
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.block1 = ResidualBlock(base, base*2, time_dim)
        self.pool1 = nn.AvgPool2d(2)
        self.block2 = ResidualBlock(base*2, base*2, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block3 = ResidualBlock(base*2 + base*2, base, time_dim)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        x1 = self.in_conv(x)
        x2 = self.block1(x1, t_emb)
        x2 = self.pool1(x2)
        x3 = self.block2(x2, t_emb)
        x3 = self.up1(x3)
        x3 = torch.cat([x3, x2], dim=1)
        x4 = self.block3(x3, t_emb)
        return self.out_conv(x4)

# ============================
# 6) Amostragem
# ============================
@torch.no_grad()
def p_sample_loop(model, n=8, T=T):  # n menor por padrão
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    
    # Usar apenas alguns passos para ser mais rápido
    steps = list(range(T-1, -1, -max(1, T//20)))
    if steps[-1] != 0:
        steps.append(0)
    
    for t_inv in steps:
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
epochs = st.sidebar.number_input("Épocas", 1, 5, 1)  # Máximo 5 épocas
n_samples = st.sidebar.slider("Amostras", 1, 16, 8)

# Inicializar estado da sessão
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_done' not in st.session_state:
    st.session_state.training_done = False

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 Demonstração do Processo")
    
    if st.button("Mostrar Processo de Difusão", key="diffuse"):
        x0, _ = next(iter(train_loader))
        x0 = x0[:4].to(device)
        
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes = axes.flatten()
        
        ts = [0, T//3, 2*T//3, T-1]
        for i, tval in enumerate(ts):
            t = torch.full((x0.size(0),), tval, device=device, dtype=torch.long)
            xt, _ = q_sample(x0, t)
            im = xt[0].detach().cpu().squeeze(0)
            im = (im + 1) / 2
            axes[i].imshow(im, cmap="gray")
            axes[i].set_title(f"Passo t={tval}")
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
    if st.button("Carregar Modelo Pré-treinado", key="load_pretrained"):
        # Criar e carregar modelo simples
        model = MiniUNet().to(device)
        st.session_state.model = model
        st.session_state.trained = True
        st.success("Modelo carregado! Agora você pode gerar amostras.")

with col2:
    st.subheader("🤖 Treinamento e Geração")
    
    if st.button("Treinar Novo Modelo", key="train"):
        with st.spinner(f"Treinando modelo por {epochs} época(s)..."):
            model = MiniUNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            progress_bar = st.progress(0)
            loss_text = st.empty()
            
            for ep in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_idx, (x0, _) in enumerate(train_loader):
                    x0 = x0.to(device)
                    t = torch.randint(0, T, (x0.size(0),), device=device)
                    xt, eps = q_sample(x0, t)
                    eps_pred = model(xt, t)
                    loss = F.mse_loss(eps_pred, eps)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Atualizar progresso
                    if batch_idx % 2 == 0:
                        progress = ((ep * len(train_loader) + batch_idx) / 
                                   (epochs * len(train_loader)))
                        progress_bar.progress(min(1.0, progress))
                
                avg_loss = total_loss / num_batches
                loss_text.text(f"Época {ep+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            st.session_state.model = model
            st.session_state.trained = True
            st.session_state.training_done = True
            
            st.success("✅ Treinamento concluído!")
            
            # Mostrar loss final
            st.metric("Loss Final", f"{avg_loss:.4f}")

# Geração de amostras
if st.session_state.trained:
    st.subheader("🎨 Gerar Novos Dígitos")
    
    col_gen1, col_gen2 = st.columns([3, 1])
    
    with col_gen2:
        if st.button("Gerar Amostras", key="generate"):
            if st.session_state.model is not None:
                with st.spinner("Gerando dígitos a partir do ruído..."):
                    samples = p_sample_loop(st.session_state.model, n=n_samples)
                    show_images_st(samples, f"Dígitos Gerados ({n_samples} amostras)", n=n_samples)
                    
                    # Opção para salvar
                    if st.download_button(
                        label="📥 Salvar Amostras",
                        data=str(samples.cpu().numpy()),
                        file_name="digitos_gerados.txt",
                        mime="text/plain"
                    ):
                        st.success("Amostras salvas!")
            else:
                st.warning("Modelo não encontrado. Treine primeiro.")
