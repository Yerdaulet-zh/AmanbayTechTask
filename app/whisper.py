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
    


class CachedMultiHeadAttentionDecoderSelf(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor,
        offset: Tensor,
    ):
        # q will always come from the bottom (from previous decoder)
        q = self.query(x)
        
        # It is essential to define the batch_indices for the case where the offset is not unique 
        # batch_indices = torch.arange(x.size(0), device=x.device, dtype=torch.int32)
        
        key = self.key(x)
        value = self.value(x)

        key_cache = torch.cat([kv_cache[:, self.n_layer, 0, ...], key], dim=1)
        value_cache = torch.cat([kv_cache[:, self.n_layer, 1, ...], value], dim=1)

        k = key_cache
        v = value_cache

        wv = self.masked_qkv_attention(q, k, v)
        return self.out(wv), key, value

    def masked_qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
    ):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.size(0), q.size(1), self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        
        qk = q @ k

        # Mask padded tokens, they deserve 0 attention score
        padding_mask = (qk == 0)
        qk.masked_fill_(padding_mask, float('-inf')) # -- more advanced, ONNX warnings
        
        # mask = padding_mask * -65504 # Smallest value for float16
        # qk = qk + mask
        
        # the model expects one token at a time
        # if mask is not None:
        #     print("qk.shape, mask.shape, n_ctx", qk.shape, mask.shape, n_ctx)
        #     qk = qk + mask[:n_ctx, :n_ctx]
        
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)




class CachedMultiHeadAttentionDecoderCross(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
    ):
        # q will always come from the bottom (from previous decoder)
        q = self.query(x)
        
        # for corss-attention
        k = n_layer_cross_k[self.n_layer, ...]
        v = n_layer_cross_v[self.n_layer, ...]
        
        wv = self.masked_qkv_attention(q, k, v)
        return self.out(wv)

    def masked_qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
    ):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.size(0), q.size(1), self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        
        qk = q @ k
        
        # the model expects one token at a time
        # if mask is not None:
        #     print("qk.shape, mask.shape, n_ctx", qk.shape, mask.shape, n_ctx)
        #     qk = qk + mask[:n_ctx, :n_ctx]
        
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        # standard encoder attention block with skip connection
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x