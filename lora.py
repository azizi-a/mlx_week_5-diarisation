import torch
from torch import nn


class LoRALayer(nn.Module):
  """Low-Rank Adaptation (LoRA) layer"""

  def __init__(self, in_dim, out_dim, rank=16, alpha=32):
    super().__init__()
    self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
    self.B = nn.Parameter(torch.zeros(rank, out_dim))
    self.alpha = alpha
    self.rank = rank

  def forward(self, x):
    return (self.alpha / self.rank) * (x @ self.A @ self.B)


def add_lora_to_linear(linear_layer, rank=16, alpha=32):
  """Add LoRA to a linear layer"""
  in_dim, out_dim = linear_layer.weight.shape
  lora = LoRALayer(in_dim, out_dim, rank, alpha)

  # Move LoRA to the same device as the linear layer
  lora = lora.to(linear_layer.weight.device)

  original_forward = linear_layer.forward

  def forward_with_lora(self, x):
    return original_forward(x) + lora(x)

  linear_layer.forward = forward_with_lora.__get__(linear_layer)
  linear_layer.lora = lora

  return linear_layer


def apply_lora_to_whisper(model, rank=16, alpha=32):
  """Apply LoRA to Whisper model attention layers"""
  lora_params = []

  # Apply to encoder
  for block in model.encoder.blocks:
    block.attn.query = add_lora_to_linear(block.attn.query, rank, alpha)
    block.attn.key = add_lora_to_linear(block.attn.key, rank, alpha)
    block.attn.value = add_lora_to_linear(block.attn.value, rank, alpha)
    block.attn.out = add_lora_to_linear(block.attn.out, rank, alpha)

    lora_params.extend(list(block.attn.query.lora.parameters()))
    lora_params.extend(list(block.attn.key.lora.parameters()))
    lora_params.extend(list(block.attn.value.lora.parameters()))
    lora_params.extend(list(block.attn.out.lora.parameters()))

  # Apply to decoder
  for block in model.decoder.blocks:
    block.attn.query = add_lora_to_linear(block.attn.query, rank, alpha)
    block.attn.key = add_lora_to_linear(block.attn.key, rank, alpha)
    block.attn.value = add_lora_to_linear(block.attn.value, rank, alpha)
    block.attn.out = add_lora_to_linear(block.attn.out, rank, alpha)

    lora_params.extend(list(block.attn.query.lora.parameters()))
    lora_params.extend(list(block.attn.key.lora.parameters()))
    lora_params.extend(list(block.attn.value.lora.parameters()))
    lora_params.extend(list(block.attn.out.lora.parameters()))

  # Freeze original parameters
  for param in model.parameters():
    param.requires_grad = False

  # Make LoRA parameters trainable
  for param in lora_params:
    param.requires_grad = True

  return model, lora_params
