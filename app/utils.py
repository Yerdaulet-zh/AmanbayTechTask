import os
import torch
import configs
import torchaudio
import numpy as np
import soundfile as sf

from typing import Tuple
from typing import KeysView
from functools import lru_cache
from transformers import WhisperTokenizerFast


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
) -> dict[dict[str: torch.Tensor]]:
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
    medium_weights: dict[dict[str: torch.Tensor]]
) -> dict[str: torch.Tensor]: 
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
    medium_weights: dict[dict[str: torch.Tensor]]
) -> dict[str: torch.Tensor]: 
    """
    Returns the extracted weights from original .pt file 
    and adjustes the keys to match with encoder block
    """
    decoder_weights = {}
    for whisper_key, decoder_key in zip(decoder_needed_keys, decoder_keys):
        decoder_weights[decoder_key] = medium_weights['model_state_dict'][whisper_key]
    return decoder_weights


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    # assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filters_path = os.path.join(base_dir, "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def compute_features(
        wave: np.array, 
        sample_rate: int,
        device: str
    ) -> torch.Tensor | dict:
    """
    Args:
        encoded_audio: Base64 encoded string of the audio file.
        sample_rate: Sample rate of the audio.
    Returns:
        Return a 1-D float32 tensor of shape (1, 80, 500) containing the features.
    """
    try:
        audio = torch.from_numpy(wave).contiguous().to(device)
        
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sample_rate, new_freq=16000
            ).float()
        else:
            audio = audio.float()

        window = torch.hann_window(400).to(device)
        stft = torch.stft(audio.float(), 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters(device, 80)
        mel_spec = filters @ magnitudes
        
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        mel = (log_spec + 4.0) / 4.0
        mel = mel.permute(1, 0)

        target = 3000
        if mel.size(0) > target:
            mel = mel[: target]
            # mel = torch.nn.functional.pad(mel, (0, 0, 0, 50), "constant", 0)
        else:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.size(0)), "constant", 0)
        mel = mel.t()
        mel = mel.unsqueeze(dim=0)
        return mel.to(torch.float16)
    except Exception as e:
        raise Exception("Error in computing features: " + str(e))


def get_audio_mel(
    audio_path: str
) -> torch.Tensor:
    wave, sr = sf.read('english_sample.wav')
    mel = compute_features(wave, sample_rate=sr)
    return mel


def update_kv_cache(
    kv_cache: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor, 
    offset: int
) -> Tuple[torch.Tensor, int]:
    kv_cache[..., 0, offset, :] = keys.squeeze(2)
    kv_cache[..., 1, offset, :] = values.squeeze(2)
    offset += 1
    return kv_cache, offset


def get_token(
    logits: torch.Tensor
) -> torch.Tensor:
    last = logits[:, -1]
    last[:, configs.suppress_nonspeech] = -torch.inf
    last = last.argmax(-1, keepdim=True)
    return last


def load_tokenizer() -> WhisperTokenizerFast:
    print("Loading Whisper Medium v3 tokenizer...")
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-medium")
    return tokenizer
