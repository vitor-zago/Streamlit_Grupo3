"""
Streamlit App for Mini-DDPM - Simplified Version
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Page config
st.set_page_config(
    page_title="Mini-DDPM Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Mini-DDPM Image Generator")
st.markdown("Generate images using a simplified Diffusion Model")

# Try to import torch with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    TORCH_AVAILABLE = True
except ImportError as e:
    st.error(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    st.stop()

# ============================
# Simplified Model Definition
# ============================

def sinusoidal_time_embedding(t, dim=64):
    """Simple time embedding function"""
    import math
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class SimpleUNet(nn.Module):
    """Simplified UNet for demo purposes"""
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, 64)
        t_emb = self.time_embed(t_emb)[:, :, None, None]
        
        # Downsample
        x1 = F.relu(self.norm1(self.conv1(x)))
        x2 = self.pool(x1)
        x2 = F.relu(self.norm2(self.conv2(x2)))
        x3 = self.pool(x2)
        
        # Upsample
        x3 = self.up(x3)
        x3 = F.relu(self.norm3(self.conv3(x3 + x2)))
        x3 = self.up(x3)
        x3 = self.conv4(x3 + x1)
        
        return x3

# ============================
# Sample Generation (Demo Mode)
# ============================

def generate_demo_samples(n_samples=16):
    """Generate random samples for demo purposes"""
    if not TORCH_AVAILABLE:
        # Generate random images for demo
        return np.random.rand(n_samples, 1, 28, 28)
    
    # Create simple noise patterns
    samples = []
    for i in range(n_samples):
        # Create different patterns
        if i % 4 == 0:
            # Vertical lines
            img = np.linspace(0, 1, 28).reshape(1, 28, 1)
            img = np.repeat(img, 28, axis=2)
        elif i % 4 == 1:
            # Horizontal lines
            img = np.linspace(0, 1, 28).reshape(28, 1, 1)
            img = np.repeat(img, 28, axis=1)
        elif i % 4 == 2:
            # Checkerboard
            x = np.arange(28)
            img = (x[:, None] // 4 + x[None, :] // 4) % 2
            img = img.reshape(1, 28, 28)
        else:
            # Random noise
            img = np.random.randn(1, 28, 28) * 0.5 + 0.5
        
        samples.append(img)
    
    samples = np.clip(np.array(samples), 0, 1)
    return samples

# ============================
# Visualization Functions
# ============================

def plot_samples(samples, title="Generated Samples"):
    """Plot grid of samples"""
    n = len(samples)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n):
        row = i // n_cols
        col = i % n_cols
        
        if samples[i].shape[0] == 1:  # Grayscale
            axes[row, col].imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        else:  # RGB
            axes[row, col].imshow(samples[i].transpose(1, 2, 0))
        
        axes[row, col].axis("off")
    
    # Hide empty subplots
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

# ============================
# Streamlit UI
# ============================

# Sidebar
st.sidebar.header("⚙️ Settings")

# Model selection
model_mode = st.sidebar.radio(
    "Model Mode",
    ["Demo Mode", "Pretrained Model"],
    help="Demo mode shows sample patterns. Pretrained requires model file."
)

# Generation parameters
n_samples = st.sidebar.slider("Number of samples", 1, 32, 16, 1)
image_size = st.sidebar.selectbox("Image size", [28, 32, 64], index=0)

# Generation button
generate_btn = st.sidebar.button("🎲 Generate Samples", type="primary")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generated Images")
    
    if generate_btn:
        with st.spinner("Generating samples..."):
            # Generate samples based on mode
            if model_mode == "Demo Mode":
                samples = generate_demo_samples(n_samples)
            else:
                # In a real app, you would load a pretrained model here
                samples = generate_demo_samples(n_samples)
                st.info("Pretrained model mode selected. In a full implementation, this would load a trained diffusion model.")
            
            # Display samples
            fig = plot_samples(samples, f"Generated Samples ({model_mode})")
            st.pyplot(fig)
            
            # Save option
            if st.button("💾 Save as PNG"):
                fig.savefig("generated_samples.png", dpi=150, bbox_inches='tight')
                st.success("Saved as generated_samples.png")
    
    else:
        st.info("Click the 'Generate Samples' button to create images")

with col2:
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### Mini-DDPM Demo
    
    This is a simplified demonstration of Diffusion Models.
    
    **Features:**
    - Generate synthetic images
    - Multiple pattern types
    - Adjustable parameters
    
    **Real implementation would:**
    1. Load trained model weights
    2. Perform actual diffusion sampling
    3. Support conditional generation
    
    **Tech Stack:**
    - PyTorch for ML
    - Streamlit for UI
    - Matplotlib for viz
    """)
    
    # Show system info
    with st.expander("System Information"):
        st.text(f"PyTorch available: {TORCH_AVAILABLE}")
        if TORCH_AVAILABLE:
            st.text(f"PyTorch version: {torch.__version__}")
        st.text(f"NumPy version: {np.__version__}")

# Add some examples
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Examples")

example_mode = st.sidebar.selectbox(
    "Quick Examples",
    ["Random", "Stripes", "Checkerboard", "Gradient"]
)

if st.sidebar.button("Load Example"):
    with col1:
        st.info(f"Loading {example_mode} example...")
        
        # Create example based on selection
        if example_mode == "Stripes":
            img = np.zeros((1, 28, 28))
            for i in range(0, 28, 4):
                img[:, :, i:i+2] = 1
        elif example_mode == "Checkerboard":
            x = np.arange(28)
            img = (x[:, None] // 4 + x[None, :] // 4) % 2
            img = img.reshape(1, 28, 28)
        elif example_mode == "Gradient":
            x = np.linspace(0, 1, 28)
            img = x[:, None] * x[None, :]
            img = img.reshape(1, 28, 28)
        else:  # Random
            img = np.random.rand(1, 28, 28)
        
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{example_mode} Example")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Mini-DDPM Demo | Educational Purpose | Built with Streamlit")
