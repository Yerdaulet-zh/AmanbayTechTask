from torch import nn
from torch import Tensor
from utils import sinusoids
from dataclasses import dataclass
from typing import Optional, Generator

import torch.nn.functional as F
import numpy as np
import torch


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int



class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: Tensor):
        # multi head attention used in the encoder
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv = self.qkv_attention(q, k, v)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.size(0), q.size(1), self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
    
