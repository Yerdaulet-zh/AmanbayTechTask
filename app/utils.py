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
