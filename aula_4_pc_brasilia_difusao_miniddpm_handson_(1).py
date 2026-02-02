import streamlit as st
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="DDPM Pro Fix", layout="wide")
T = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODELO REFORÇADO ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.mlp = nn.Linear(t_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x, t):
        h = F.silu(self.bn(self.conv1(x)))
        h = h + self.mlp(t)[:, :, None, None]
        h = F.silu(self.bn(self.conv2(h)))
        return h

class FullUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(128), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.inc = nn.Conv2d(1, 64, 3, padding=1)
        self.down1 = ConvBlock(64, 128, 256)
        self.down2 = ConvBlock(128, 256, 256)
        self.mid = ConvBlock(256, 256, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = ConvBlock(256, 64, 256)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = F.max_pool2d(x2, 2)
        x4 = self.down2(x3, t)
        x5 = F.max_pool2d(x4, 2)
        x6 = self.mid(x5, t)
        x7 = self.up1(x6)
        x = torch.cat([x7, x4], dim=1)
        x = self.up_conv1(x, t)
        x = self.up2(x)
        return self.outc(x)

# --- INICIALIZAÇÃO ---
if "model" not in st.session_state:
    st.session_state.model = FullUNet().to(device)
if "loss" not in st.session_state:
    st.session_state.loss = []

# --- PARÂMETROS DIFUSÃO ---
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# --- SIDEBAR E TREINO ---
st.sidebar.title("Controlo")
if st.sidebar.button("🚀 Treinar (Reforçado)"):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2-1)])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tr)
    # Aumentar o subset é vital para sair do cinzento
    loader = DataLoader(Subset(ds, range(10000)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(st.session_state.model.parameters(), lr=5e-4)
    
    st.session_state.model.train()
    for ep in range(15): # Mínimo 15 épocas
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            a_t = alpha_bar[t].view(-1, 1, 1, 1)
            xt = torch.sqrt(a_t)*x0 + torch.sqrt(1-a_t)*noise
            
            pred_noise = st.session_state.model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
            
            opt.zero_grad(); loss.backward(); opt.step()
        st.session_state.loss.append(loss.item())
        st.sidebar.write(f"Ep {ep}: Loss {loss.item():.4f}")

if st.session_state.loss:
    st.sidebar.line_chart(st.session_state.loss)

# --- GERAÇÃO ---
if st.button("✨ Gerar Imagens"):
    st.session_state.model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    cols = st.columns(5)
    steps = [499, 300, 150, 50, 0]
    idx = 0
    
    for t_inv in range(T-1, -1, -1):
        t_batch = torch.full((1,), t_inv, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_n = st.session_state.model(x, t_batch)
            a_t = alpha_bar[t_inv]
            # Previsão x0 corrigida
            x0_p = (x - (1-a_t).sqrt() * pred_n) / a_t.sqrt()
            x0_p = torch.clamp(x0_p, -1, 1)
            
            if t_inv > 0:
                z = torch.randn_like(x)
                x = 1/alpha[t_inv].sqrt() * (x - beta[t_inv]/(1-alpha_bar[t_inv]).sqrt() * pred_n) + beta[t_inv].sqrt() * z
            else:
                x = x0_p
        
        if t_inv in steps:
            with cols[idx]:
                st.image((x.squeeze().cpu().numpy()+1)/2, caption=f"xt (t={t_inv})", use_container_width=True)
                st.image((x0_p.squeeze().cpu().numpy()+1)/2, caption="Previsão x0", use_container_width=True)
            idx += 1
