# -*- coding: utf-8 -*-
"""
Código corrigido do Mini-DDPM para MNIST
Versão simplificada para execução local
"""

# ============================
# 1) Setup do ambiente
# ============================
import math
import os
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print(f"PyTorch version: {torch.__version__}")

plt.rcParams["figure.dpi"] = 120

# ============================
# 2) Dataset MNIST (imagens 28x28)
# ============================
# Usando o MNIST padrão do torchvision
transform = transforms.Compose([
    transforms.ToTensor(),                  # [0,1]
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1]
])

# Carregar MNIST
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Criar subset para treinamento mais rápido
subset_size = 10000
indices = np.random.choice(len(train_ds), size=subset_size, replace=False)
train_subset = Subset(train_ds, indices)

# DataLoader
batch_size = 128
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

print(f"\n✓ MNIST carregado! {len(train_ds)} imagens no total")
print(f"✓ Usando subset de {len(train_subset)} imagens")

# ============================
# Função para visualizar imagens (tons de cinza)
# ============================
def show_images(x, title="Amostras", n=16):
    x = x[:n].detach().cpu()
    x = (x + 1) / 2  # [-1,1] -> [0,1]
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    axes = axes.flatten()
    
    for i in range(n):
        # Para MNIST: [1, 28, 28] -> [28, 28]
        img = x[i].squeeze(0)  # Remove channel dimension
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Testar visualização
x0, y0 = next(iter(train_loader))
show_images(x0, "MNIST - Dígitos")
print(f"Batch shape: {x0.shape}")
print(f"Labels: {y0[:16].tolist()}")

# ============================
# 3) Noise Schedule (β, α, ᾱ)
# ============================
T = 200  # passos

beta_start = 1e-4
beta_end = 0.02
beta = torch.linspace(beta_start, beta_end, T)  # (T,)

alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)         # (T,)

beta = beta.to(device)
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

# Visualizar noise schedule
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(beta.detach().cpu().numpy())
plt.title("beta_t (noise schedule)")
plt.xlabel("Passo t")
plt.ylabel("Beta")

plt.subplot(1, 2, 2)
plt.plot(alpha_bar.detach().cpu().numpy())
plt.title("alpha_bar_t")
plt.xlabel("Passo t")
plt.ylabel("Alpha_bar")
plt.tight_layout()
plt.show()

print("beta[0], beta[-1]:", float(beta[0]), float(beta[-1]))
print("alpha_bar[0], alpha_bar[-1]:", float(alpha_bar[0]), float(alpha_bar[-1]))

# ============================
# 4) Processo direto: q(x_t | x_0)
# ============================
def q_sample(x0, t, noise=None):
    """
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    # Garantir que t seja tensor e tenha dimensões adequadas
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)
    
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# Visualizar processo de difusão
x0_batch = x0.to(device)
ts = [0, 20, 60, 120, 199]
imgs = []

plt.figure(figsize=(15, 3))
for i, tval in enumerate(ts):
    t = torch.full((x0_batch.size(0),), tval, device=device, dtype=torch.long)
    xt, _ = q_sample(x0_batch, t)
    
    plt.subplot(1, len(ts), i+1)
    im = xt[0].detach().cpu()
    im = (im + 1) / 2
    plt.imshow(im.squeeze(0), cmap="gray")
    plt.title(f"t={tval}")
    plt.axis("off")

plt.suptitle("Processo direto: adicionando ruído (x0 -> xt)")
plt.tight_layout()
plt.show()

# ============================
# 5.1) Embedding de tempo (sinusoidal)
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

# Testar embedding
t_test = torch.tensor([0, 10, 199], device=device)
emb = sinusoidal_time_embedding(t_test, dim=64)
print("Embedding shape:", emb.shape)

# ============================
# 5.2) U-Net pequena para MNIST
# ============================
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

        self.down1 = Down(base, base*2, time_dim)      # 28 -> 14
        self.down2 = Down(base*2, base*4, time_dim)    # 14 -> 7

        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.bot2 = ResidualBlock(base*4, base*4, time_dim)

        self.up2 = Up(base*4 + base*4, base*2, time_dim)  # 7 -> 14
        self.up1 = Up(base*2 + base*2, base, time_dim)    # 14 -> 28

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

# Instanciar modelo
model = MiniUNet(time_dim=64, base=32).to(device)

# Testar forward pass
x0_test, _ = next(iter(train_loader))
x0_test = x0_test.to(device)
t_test = torch.randint(0, T, (x0_test.size(0),), device=device)
eps_pred = model(x0_test, t_test)
print(f"x0 shape: {x0_test.shape}, eps_pred shape: {eps_pred.shape}")

# Contar parâmetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total de parâmetros do modelo: {total_params:,}")

# ============================
# 6) Loop de treinamento
# ============================
lr = 2e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def train_one_epoch(model, loader):
    model.train()
    losses = []
    
    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(device)
        
        # Amostrar t uniformemente
        t = torch.randint(0, T, (x0.size(0),), device=device, dtype=torch.long)
        
        # Adicionar ruído (processo direto)
        xt, eps = q_sample(x0, t)
        
        # Prever ruído
        eps_pred = model(xt, t)
        
        # Calcular loss
        loss = F.mse_loss(eps_pred, eps)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Log a cada 50 batches
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")
    
    return float(np.mean(losses))

# Treinar por algumas épocas
epochs = 3
history = []

print("\n" + "="*50)
print("INICIANDO TREINAMENTO")
print("="*50)

for ep in range(1, epochs + 1):
    print(f"\nEpoch {ep}/{epochs}")
    mean_loss = train_one_epoch(model, train_loader)
    history.append(mean_loss)
    print(f"Epoch {ep} concluída - Loss média: {mean_loss:.4f}")

# Plotar histórico de loss
plt.figure()
plt.plot(history, marker="o", linestyle="-")
plt.title("Loss por época (MSE do ruído ε)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()

# Salvar checkpoint
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/mini_ddpm_mnist.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epochs,
    "loss": history[-1],
    "T": T,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "alpha_bar": alpha_bar.cpu()
}, cpt_path)
print(f"\n✓ Checkpoint salvo em: {ckpt_path}")

# ============================
# 7) Amostragem (gerar novas imagens)
# ============================
@torch.no_grad()
def p_sample(model, x, t):
    """Um passo de sampling reverso"""
    model.eval()
    
    # Prever ruído
    eps_pred = model(x, t)
    
    # Calcular x0 aproximado
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)
    x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
    
    # Calcular média e variância para o próximo passo
    a = alpha[t].view(-1, 1, 1, 1)
    a_bar_prev = alpha_bar[t-1].view(-1, 1, 1, 1) if t > 0 else torch.ones_like(a_bar)
    
    # Coeficientes
    mean_coef1 = torch.sqrt(a) * (1 - a_bar_prev) / (1 - a_bar)
    mean_coef2 = torch.sqrt(a_bar_prev) * (1 - a) / (1 - a_bar)
    mean = mean_coef1 * x + mean_coef2 * x0_hat
    
    if t > 0:
        variance = (1 - a_bar_prev) / (1 - a_bar) * (1 - a)
        std = torch.sqrt(variance)
        noise = torch.randn_like(x)
        return mean + std * noise
    else:
        return mean

@torch.no_grad()
def p_sample_loop(model, n_samples=16, return_frames=False):
    """Loop completo de sampling reverso"""
    model.eval()
    
    # Começar com ruído puro
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    
    frames = []
    
    # Reversão do processo (T-1 até 0)
    for t_inv in range(T-1, -1, -1):
        t = torch.full((n_samples,), t_inv, device=device, dtype=torch.long)
        x = p_sample(model, x, t)
        
        if return_frames and t_inv in [T-1, int(T*0.75), int(T*0.5), int(T*0.25), 0]:
            frames.append(x.clone())
    
    x = torch.clamp(x, -1, 1)
    
    if return_frames:
        return x, frames
    return x

# Gerar amostras
print("\n" + "="*50)
print("GERANDO AMOSTRAS")
print("="*50)

samples, frames = p_sample_loop(model, n_samples=16, return_frames=True)

# Mostrar amostras finais
show_images(samples, "Amostras geradas pelo modelo", n=16)

# Mostrar evolução da geração
print("\nEvolução da geração (frames em diferentes passos):")
plt.figure(figsize=(15, 8))
step_names = ["Ruído puro (t=199)", "t=150", "t=100", "t=50", "Final (t=0)"]

for i, (frame, step_name) in enumerate(zip(frames, step_names)):
    plt.subplot(2, 3, i+1)
    
    # Pegar as primeiras 6 imagens do frame
    grid_img = torch.cat([(img.detach().cpu().squeeze(0) + 1) / 2 for img in frame[:6]], dim=2)
    plt.imshow(grid_img, cmap="gray")
    plt.title(step_name)
    plt.axis("off")

plt.suptitle("Evolução da geração (difusão reversa)")
plt.tight_layout()
plt.show()

# ============================
# 8) Testar interpolação entre dígitos
# ============================
@torch.no_grad()
def interpolate(model, z1, z2, steps=5):
    """Interpolação linear no espaço latente"""
    model.eval()
    alphas = torch.linspace(0, 1, steps, device=device)
    
    interpolated_samples = []
    
    for alpha in alphas:
        z = alpha * z1 + (1 - alpha) * z2
        x = z.clone()
        
        # Processo de sampling reverso
        for t_inv in range(T-1, -1, -1):
            t = torch.full((z1.size(0),), t_inv, device=device, dtype=torch.long)
            x = p_sample(model, x, t)
        
        interpolated_samples.append(x)
    
    return interpolated_samples

# Gerar duas sementes de ruído diferentes
z1 = torch.randn(1, 1, 28, 28, device=device)
z2 = torch.randn(1, 1, 28, 28, device=device)

# Interpolar
interpolated = interpolate(model, z1, z2, steps=7)

# Visualizar interpolação
print("\nInterpolação entre duas sementes de ruído:")
fig, axes = plt.subplots(1, 7, figsize=(15, 3))

for i, sample in enumerate(interpolated):
    img = sample[0].detach().cpu().squeeze(0)
    img = (img + 1) / 2
    axes[i].imshow(img, cmap="gray")
    axes[i].axis('off')
    axes[i].set_title(f"α={i/6:.1f}")

plt.suptitle("Interpolação no espaço latente")
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("EXECUÇÃO CONCLUÍDA!")
print("="*50)
print(f"Modelo treinado por {epochs} épocas")
print(f"Loss final: {history[-1]:.4f}")
print(f"Amostras salvas e visualizadas")
