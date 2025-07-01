import torch
import numpy as np

from typing import KeysView


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def load_original_whisper_weights(
    file_path: str,
    device: str,
) -> dict[dict[str: torch.tensor]]:
    """
    Load whisper weights to the system
    """
    weights = torch.load(
        f=file_path,
        weights_only=True,
        map_location=device
    )
    return weights


def get_whisper_encoder_keys(
    encoder_keys: KeysView[str]
) -> list[str]:
    """
    Returns: list[str] 
    Whisper state dict keys which will be used to extract 
    from model.pt only required weights for encoder block 
    (the keys are adjusted for only original whisper layer names). 
    
    Note:
        Some decoder layers are pre-computed at encoder layer, 
        which can be directly applied in decoder for cross-attention.
    """
    
    encoder_needed_keys = []
    
    for key in encoder_keys:
        if key.startswith('decoder'):
            key = 'decoder.blocks' + key[len('decoder'):]
            encoder_needed_keys.append(key)
            continue
        
        key = 'encoder.' + key
        encoder_needed_keys.append(key)
    return encoder_needed_keys


def get_whisper_encoder_weigths(
    encoder_keys: KeysView[str],
    encoder_needed_keys: list[str],
    medium_weights: dict[dict[str: torch.tensor]]
) -> dict[str: torch.tensor]: 
    """
    Returns the extracted weights from original .pt file 
    and adjustes the keys to match with encoder block
    """
    encoder_weights = {}
    for whisper_key, encoder_key in zip(encoder_needed_keys, encoder_keys):
        encoder_weights[encoder_key] = medium_weights['model_state_dict'][whisper_key]
    return encoder_weights


def get_whisper_decoder_keys(
    decoder_keys: KeysView[str]
) -> list[str]:
    """
    Returns: list[str] 
    Whisper state dict keys which will be used to extract 
    from model.pt only required weights for decoder block 
    (the keys are adjusted for only original whisper layer names).
    """
    
    decoder_needed_keys = []
    
    for key in decoder_keys:
        key = 'decoder.' + key
        decoder_needed_keys.append(key)
    return decoder_needed_keys


def get_whisper_decoder_weigths(
    decoder_keys: KeysView[str],
    decoder_needed_keys: list[str],
    medium_weights: dict[dict[str: torch.tensor]]
) -> dict[str: torch.tensor]: 
    """
    Returns the extracted weights from original .pt file 
    and adjustes the keys to match with encoder block
    """
    decoder_weights = {}
    for whisper_key, decoder_key in zip(decoder_needed_keys, decoder_keys):
        decoder_weights[decoder_key] = medium_weights['model_state_dict'][whisper_key]
    return decoder_weights

