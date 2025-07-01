from torch import nn
from torch import Tensor
from .utils import sinusoids
from dataclasses import dataclass

import torch.nn.functional as F
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
    


class CachedResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.attn = CachedMultiHeadAttentionDecoderSelf(n_state, n_head, n_layer)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = CachedMultiHeadAttentionDecoderCross(n_state, n_head, n_layer)
        self.cross_attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        kv_cache: Tensor,
        n_layer_cross_k,
        n_layer_cross_v
    ):
        # decoder attn and cross-attn block with skip connection
        x1, k, v = self.attn(self.attn_ln(x), kv_cache)
        x = x + x1
        x = x + self.cross_attn(self.cross_attn_ln(x), n_layer_cross_k, n_layer_cross_v)
        x = x + self.mlp(self.mlp_ln(x))
        return x, k, v
    


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layers: int, encoder_x: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.encoder_x = encoder_x

        # encoder
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layers)]
        )
        self.ln_post = nn.LayerNorm(n_state)
        
        # decoder
        self.decoder = nn.ModuleList(
            [
                CachedResidualAttentionBlock(n_state, n_head, n_layer)
                for n_layer in range(n_layers)
            ]
        )

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x[0].size() == self.positional_embedding.size(), "incorrect audio shape"
        # x = x + self.positional_embedding
        
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        
        if self.encoder_x:
            return x
        
        ###   DECODER   ###
        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.decoder:
            n_layer_cross_k_list.append(block.cross_attn.key(x))
            n_layer_cross_v_list.append(block.cross_attn.value(x))
        audio_features = torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)
        return (audio_features[0].permute(1, 0, 2, 3), audio_features[1].permute(1, 0, 2, 3))



class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layers: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                CachedResidualAttentionBlock(n_state, n_head, n_layer)
                for n_layer in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(n_state)

        # mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        # self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, kv_cache: Tensor, n_layer_cross_k: Tensor, n_layer_cross_v: Tensor, offset: Tensor):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        """

        # (b_size, n_layers, audio_lenght, d_model)
        n_layer_cross_k = n_layer_cross_k.permute(1, 0, 2, 3)
        n_layer_cross_v = n_layer_cross_v.permute(1, 0, 2, 3)
        
        # offset = kv_cache[0].size(1) if len(kv_cache) > 0 else 0
        
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset] # We always expect a single token at a time in the batch 
        )

        keys = []
        values = []
        for block in self.blocks:
            x, k, v = block(x, kv_cache, n_layer_cross_k, n_layer_cross_v)
            keys.append(k)
            values.append(v)
        
        x = self.ln(x)
        logits = x @ torch.transpose(self.token_embedding.weight, 0, 1)
        keys, values = torch.stack((keys), dim=0), torch.stack((values), dim=0)
        return logits, keys.permute(1, 0, 2, 3), values.permute(1, 0, 2, 3)