# WGAN-GP - Wasserstein GAN with Gradient Penalty

PyTorch implementation of WGAN-GP based on the paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et al. This implementation uses the DCGAN architecture with Wasserstein loss and gradient penalty for stable, high-quality image generation.

## Overview

WGAN-GP is an improvement over the original Wasserstein GAN that replaces weight clipping with a gradient penalty term. This modification addresses training instability issues and enables stable training of a wide variety of GAN architectures with minimal hyperparameter tuning.

### Why WGAN-GP?

The original WGAN used weight clipping to enforce the Lipschitz constraint, which led to several problems:
- **Capacity Underuse**: Critics learn overly simple functions
- **Gradient Issues**: Either exploding or vanishing gradients
- **Convergence Problems**: Sometimes fails to converge or generates low-quality samples

WGAN-GP solves these issues by penalizing the norm of the gradient of the critic with respect to its input, leading to smoother convergence and higher-quality sample generation.

## Key Improvements Over WGAN

| Feature | WGAN | WGAN-GP |
|---------|------|---------|
| **Lipschitz Constraint** | Weight clipping [-0.01, 0.01] | Gradient penalty |
| **Optimizer** | RMSprop | Adam (works well) |
| **Learning Rate** | 5e-5 | 1e-4 |
| **Batch Normalization** | Can be used | Not in critic |
| **Training Stability** | Moderate | Excellent |
| **Convergence** | Slower, sometimes fails | Faster, more reliable |
| **Sample Quality** | Good | Better |

## Architecture

### Generator
- **Input**: 100-dimensional noise vector
- **Architecture**: DCGAN-style
  - Transposed convolutions with stride=2
  - **Batch normalization** in all hidden layers
  - ReLU activations
  - Tanh output activation
- **Output**: 64x64 images
- **Features**: 16 base feature maps (configurable)

### Critic
- **Input**: 64x64 images
- **Architecture**: DCGAN-style discriminator
  - Strided convolutions (stride=2)
  - **Instance normalization** (not batch norm!)
  - LeakyReLU(0.2) activations
  - No sigmoid at output
- **Output**: Unbounded scalar score
- **Features**: 16 base feature maps (configurable)

## Gradient Penalty Explained

The gradient penalty enforces the Lipschitz constraint by penalizing deviations from unit gradient norm:

```python
# 1. Create interpolated images between real and fake
alpha = torch.rand((batch_size, 1, 1, 1))
interpolated = real * alpha + fake * (1 - alpha)

# 2. Get critic scores for interpolated images
mixed_scores = critic(interpolated)

# 3. Compute gradient of scores w.r.t. interpolated images
gradient = autograd.grad(
    outputs=mixed_scores,
    inputs=interpolated,
    grad_outputs=torch.ones_like(mixed_scores),
    create_graph=True,
    retain_graph=True,
)[0]

# 4. Penalize deviation from gradient norm of 1
gradient_norm = gradient.norm(2, dim=1)
gradient_penalty = ((gradient_norm - 1) ** 2).mean()

# 5. Add to critic loss with weight λ
loss_critic = -(critic_real - critic_fake) + λ * gradient_penalty
```

The optimal 1-Lipschitz critic function has unit gradient norm almost everywhere under the real and generated distributions.

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
tensorboard
tqdm
```

## Installation

```bash
pip install torch torchvision tensorboard tqdm
```

## Project Structure

```
.
├── model.py           # Critic and Generator architectures
├── train.py           # WGAN-GP training script
├── utils.py           # Gradient penalty, checkpointing utilities
├── README.md          # This file
├── dataset/           # MNIST dataset (auto-downloaded)
└── logs/              # TensorBoard logs
    └── GAN_MNIST/
        ├── real/      # Real images
        └── fake/      # Generated images
```

## Usage

### Training

```bash
python train.py
```

### Testing Architecture

```bash
python model.py
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir=logs/GAN_MNIST
```

Open `http://localhost:6006` to view training progress, real vs generated images, and loss curves.

### Saving and Loading Checkpoints

The implementation includes checkpoint utilities:

```python
# Save checkpoint
save_checkpoint({
    'gen': gen.state_dict(),
    'critic': critic.state_dict(),
}, filename="my_checkpoint.pth.tar")

# Load checkpoint
load_checkpoint(checkpoint, gen, critic)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Higher than WGAN (5e-5) |
| Batch Size | 64 | Standard size |
| Image Size | 64x64 | Output resolution |
| Noise Dimension | 100 | Input latent vector size |
| Epochs | 100 | Recommended for good results |
| Critic Iterations | 5 | Critic updates per generator update |
| Lambda (λ) | 10 | Gradient penalty coefficient |
| Optimizer | Adam | Both generator and critic |
| Beta1 | 0.0 | Adam beta1 (0.0 or 0.5 works) |
| Beta2 | 0.9 | Adam beta2 |
| Features | 16 | Base feature maps (memory efficient) |

## Loss Functions

### Critic Loss
```python
loss_critic = -(E[critic(real)] - E[critic(fake)]) + λ * gradient_penalty
```

The critic maximizes the difference between real and fake scores while maintaining smooth gradients.

### Generator Loss
```python
loss_gen = -E[critic(fake)]
```

The generator maximizes the critic's score for generated images.

## Training Procedure

```python
for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        # Train Critic 5 times
        for _ in range(CRITIC_ITERATIONS):
            1. Generate fake images from noise
            2. Compute critic scores for real and fake
            3. Compute gradient penalty on interpolated images
            4. Compute total critic loss
            5. Update critic parameters
            # NO weight clipping!
        
        # Train Generator once
        1. Generate fake images from noise
        2. Compute critic score for fakes
        3. Compute generator loss
        4. Update generator parameters
```

## Key Implementation Details

### 1. No Batch Normalization in Critic

Batch normalization is not used in the critic because it maps a batch of inputs to a batch of outputs, but we need to compute gradients of each output with respect to its individual input.

```python
# Critic uses Instance Normalization
nn.InstanceNorm2d(out_channels, affine=True)

# Generator still uses Batch Normalization
nn.BatchNorm2d(out_channels)
```

### 2. retain_graph=True

Required when computing the gradient penalty because the computational graph is needed for both the gradient penalty computation and the backward pass:

```python
loss_critic.backward(retain_graph=True)
```

### 3. Lambda (λ) Parameter

The penalty coefficient λ is typically set to 10 for all experiments. This value works well across different datasets and architectures.

### 4. Adam with β₁=0.0

Using β₁=0.0 or 0.5 helps prevent momentum from causing training instability. The original paper used β₁=0.0 and β₂=0.9.

## Dataset Support

**Default**: MNIST (1 channel, grayscale)
- Auto-downloads on first run
- Resized to 64x64
- Normalized to [-1, 1]

**For CelebA or RGB datasets**:
```python
# In train.py, comment MNIST and uncomment:
dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)

# Change hyperparameters:
CHANNELS_IMG = 3
FEATURES_CRITIC = 64  # Increase for larger/color images
FEATURES_GEN = 64
```

## Advantages of WGAN-GP

✅ **Superior Stability**: Trains reliably without careful hyperparameter tuning

✅ **No Mode Collapse**: Gradient penalty prevents collapse to single modes

✅ **Better Sample Quality**: Produces higher quality images than WGAN

✅ **Meaningful Losses**: Loss values correlate with generation quality

✅ **Faster Convergence**: Converges faster than weight clipping

✅ **Architectural Flexibility**: Successfully trains diverse architectures including 101-layer ResNets

✅ **No Weight Constraints**: Removes the need for weight clipping

## Training Tips

### Convergence Time
WGAN-GPs take significant time to converge - even on MNIST, about 50 epochs are needed to see decent results. For best quality:
- Train for 100+ epochs on MNIST
- Train for 50-200 epochs on CelebA
- Be patient - quality improves gradually

### Hyperparameter Tuning

**Lambda (λ)**:
- Default: 10 (works for most cases)
- Toy problems: Try 0.1-1.0
- Complex datasets: 10-20

**Critic Iterations**:
- Start with 5 (standard)
- Increase to 10 for very stable training
- Decrease to 3 if training is too slow

**Learning Rate**:
- 1e-4 is a good default
- Try 5e-5 for more stable but slower training
- Try 2e-4 for faster but potentially less stable training

**Feature Maps**:
- 16 for memory-constrained environments
- 64 for standard training (better quality)
- 128 for high-quality generation (more memory)

## Common Issues & Solutions

### Issue: Slow Convergence
**Solutions**:
- Increase learning rate to 2e-4
- Reduce CRITIC_ITERATIONS to 3
- Increase FEATURES_CRITIC and FEATURES_GEN
- Train longer (100+ epochs)

### Issue: Low Sample Quality
**Solutions**:
- Train for more epochs (be patient!)
- Increase model capacity (FEATURES_CRITIC/GEN)
- Verify gradient penalty is computed correctly
- Check that instance norm is used in critic

### Issue: Memory Issues
**Solutions**:
- Reduce BATCH_SIZE (try 32)
- Reduce FEATURES_CRITIC and FEATURES_GEN (try 8 or 16)
- Use gradient checkpointing
- Use mixed precision training

### Issue: Training Instability
**Solutions**:
- Verify no batch normalization in critic
- Check that retain_graph=True is set
- Ensure instance normalization is used
- Try β₁=0.5 instead of 0.0

## Monitoring Training

Unlike standard GANs, WGAN-GP provides interpretable metrics:

- **Critic Loss**: Should stabilize (becomes more negative over time)
- **Generator Loss**: Should decrease (becomes more negative)
- **Sample Quality**: Should improve consistently with training time
- **Loss Correlation**: Lower (more negative) losses typically indicate better quality
