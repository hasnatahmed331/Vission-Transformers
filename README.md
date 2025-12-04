# Vision Transformer (ViT) & Cross-ViT with T2T

This repository provides PyTorch implementations of **Vision Transformer (ViT)** and **Cross-ViT**, featuring optional **Tokens-to-Token (T2T)** modules for enhanced patch embedding.

---

## Table of Contents

- [Vision Transformer (ViT)](#vision-transformer-vit)
- [Cross-ViT](#cross-vit)
- [Tokens-to-Token (t2t) Module](#tokens-to-token-t2t-module)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Running the Demo](#running-the-demo)
  - [Usage for Training](#usage-for-training)

---

## Vision Transformer (ViT)

The **Vision Transformer (ViT)** applies a pure transformer architecture directly to sequences of image patches.

### Architecture

- **Patch Embedding**  
  The input image is split into fixed-size patches (e.g., 16×16). Each patch is linearly embedded into a flat vector.

  - **Standard:**  
    Uses a `Conv2d` layer with kernel size and stride equal to the patch size.
  - **T2T (Optional):**  
    Can be configured to use a T2T module for progressive tokenization (see [Tokens-to-Token (T2T) Module](#tokens-to-token-t2t-module)).

- **Transformer Encoder**  
  A stack of standard Transformer blocks, each consisting of:
  - Multi-Head Self-Attention (**MSA**)
  - Multi-Layer Perceptron (**MLP**)
  - Layer Normalization (**LN**) before each block and residual connections after.

- **CLS Token**  
  A learnable class token is prepended to the sequence of patch embeddings.  
  Its final state serves as the image representation for classification.

- **Positional Embeddings**  
  Learnable embeddings are added to patch embeddings to retain positional information.

- **File:**  
  `vit.py`

---

## Cross-ViT

**Cross-ViT** is a dual-branch transformer architecture designed to learn multi-scale features. It processes image patches of different sizes (e.g., small and large) in separate branches and fuses them using cross-attention.

### Architecture

- **Dual Branches**
  - **Small Branch:**
    - Processes smaller patches (fine-grained details)
    - Uses fewer embedding dimensions
  - **Large Branch:**
    - Processes larger patches (coarse-grained information)
    - Uses larger embedding dimensions

- **Cross-Attention Fusion**
  - The `[CLS]` token from one branch serves as a query to attend to the patch tokens of the other branch.
  - This allows efficient information exchange between the two scales **without** the quadratic cost of full self-attention across all tokens.

- **Multi-Scale Block**
  - Contains a stack of transformer blocks for each branch.
  - Followed by a Cross-Attention Fusion module.

- **Patch Embedding**
  - Supports both standard convolutional patch embedding and T2T-based embedding for each branch **independently**.

- **File:**  
  `cross_vit.py`

---

## Tokens-to-Token (T2T) Module

The **Tokens-to-Token (T2T)** module is an advanced patch embedding mechanism that models local structural information better than a simple linear projection.

### Mechanism

It progressively reduces the spatial dimensions of the image (or feature map) while increasing the channel depth (token dimension) through an iterative process:

1. **Unfold / Conv**  
   Collects local neighboring pixels/tokens into a patch.

2. **Transformer**  
   Applies a small transformer block to model relationships within the local patch.

3. **Reshape**  
   Reshapes the output back to a 2D feature map for the next iteration.

This **"Soft Split"** strategy allows the model to learn relationships between surrounding tokens iteratively, preserving local structure.

- **File:**  
  `t2t_layer.py`

---

## How to Run

The `main.py` script provided in this repository is a **demonstration script**. It shows how to instantiate the models and perform a forward pass with random data to verify that the architectures are working correctly.

### Prerequisites

- Python 3.x  
- PyTorch  
- `einops` (for tensor manipulations)

---

### Usage for Training

These files contain model definitions that are ready to be integrated into your own training loop:

- You can **import `ModelFactory`** or the **model classes directly**.
- The models are designed to plug into a **standard PyTorch training setup**.
- Below is an example showing **how to instantiate a model** and **use it in a typical PyTorch training loop**.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from model_factory import ModelFactory

# 1. Instantiate the model (e.g., Cross-ViT)
model = ModelFactory.build(
    model_type="crossvit",
    img_size=224,                    # Standard ImageNet size
    num_classes=1000,                # Number of classes for your dataset
    cross_patch_sizes=(12, 16),      # Example patch sizes
    cross_dims=(192, 384),           # Embedding dimensions
    cross_heads=(6, 12),             # Number of attention heads
    cross_depth=[(1, 2, 1), (1, 2, 1)],  # Depth configuration
    cross_mlp_ratio=(3.0, 3.0, 1.0)
)

# 2. Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 4. Dummy Training Loop Step
inputs = torch.randn(8, 3, 224, 224).to(device)   # Batch of images
labels = torch.randint(0, 1000, (8,)).to(device)  # Batch of labels

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, labels)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training step complete. Loss: {loss.item()}")
