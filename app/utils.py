import torch

from typing import KeysView


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

