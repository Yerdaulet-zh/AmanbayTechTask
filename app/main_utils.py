import os
import torch
import datetime
import requests
import pandas as pd
import torch.nn as nn

from transformers import WhisperTokenizerFast

from utils import (
    load_original_whisper_weights,
    get_whisper_encoder_keys,
    get_whisper_encoder_weigths,
    get_whisper_decoder_keys,
    get_whisper_decoder_weigths,
    update_kv_cache,
    get_token
)


from whisper import (
    AudioEncoder, 
    TextDecoder,
    ModelDimensions
)


def load_tokenizer() -> WhisperTokenizerFast:
    print("Loading Whisper Medium v3 tokenizer...")
    tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-medium")
    return tokenizer


def ensure_directory(
    directory: str='model/'
) -> None:
    if not os.path.isdir(directory):
        print(f"The expected directory '{directory}' does not exist.\nCreating it now...")
        os.makedirs(directory)
        print("Directory created successfully.")


def ensure_model(
    model_path: str, 
    medium_model_url: str
)-> None:
    if not os.path.isfile(model_path):
        print(f"The expected file '{model_path}' does not exist.\nDownloading it now...")
        response = requests.get(url=medium_model_url, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download the model file: {response.status_code}")
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Download completed.")


def load_model(
    model_path: str, 
    device: str
) -> tuple[nn.Module, nn.Module]:
    print("Loading Whisper Medium v3 in float16...")

    medium_weights = load_original_whisper_weights(
        file_path=model_path, 
        device=device
    )
    dims = ModelDimensions(**medium_weights['dims'])

    encoder = AudioEncoder(
        n_mels=dims.n_mels,
        n_ctx=dims.n_audio_ctx,
        n_state=dims.n_audio_state,
        n_head=dims.n_audio_head,
        n_layers=dims.n_audio_layer,
    )

    decoder = TextDecoder(
        n_vocab=dims.n_vocab,
        n_ctx=dims.n_text_ctx,
        n_state=dims.n_text_state,
        n_head=dims.n_text_head,
        n_layers=dims.n_text_layer 
    )

    encoder_keys = encoder.state_dict()

    encoder_needed_keys = get_whisper_encoder_keys(encoder_keys)
    encoder_weights = get_whisper_encoder_weigths(
        encoder_keys,
        encoder_needed_keys, 
        medium_weights
    )
    encoder.load_state_dict(encoder_weights)
    encoder = encoder.to(device).half()
    encoder = encoder.eval()

    decoder_keys = decoder.state_dict()

    decoder_needed_keys = get_whisper_decoder_keys(decoder_keys)
    decoder_weights = get_whisper_decoder_weigths(
        decoder_keys, 
        decoder_needed_keys, 
        medium_weights
    )
    decoder.load_state_dict(decoder_weights)
    decoder = decoder.to(device).half()
    decoder = decoder.eval()

    print("Model Encoder and Decoder blocks are loaded successfully.\nNow the model is ready for inference.")
    return encoder, decoder


def run_inference(
    encoder: nn.Module,
    decoder: nn.Module, 
    mel: torch.Tensor,
    tokenizer: WhisperTokenizerFast,
    device: str,
    output_csv_file_path: str,
    max_token_sequence: int
) -> None:
    
    assert 1 <= max_token_sequence <= 100, "max_token_sequence must be between 1 and 100"

    print("Running inference...")
    
    offset = 0
    b_size = mel.size(0)
    max_token_sequence = max_token_sequence
    tokens = torch.tensor([[50258, 50259, 50359, 50363]] * b_size, dtype=torch.int32).to(device)
    kv_cache = torch.zeros((b_size, 24, 2, max_token_sequence, 1024), dtype=torch.half, device=device)

    
    with torch.no_grad():
        n_layer_cross_k, n_layer_cross_v = encoder(mel)
        
        # Special tokens
        for token in tokens[0]:
            logits, keys, values = decoder(
                token.unsqueeze(0).unsqueeze(0),
                kv_cache,
                n_layer_cross_k,
                n_layer_cross_v,
                offset
            )
            kv_cache, offset = update_kv_cache(kv_cache, keys, values, offset)
            last = get_token(logits)
        tokens = torch.cat([tokens, last], dim=-1)
        
        
        # Start of auto-regressiveness
        for iteration in range(max_token_sequence-len(tokens)):
            logits, keys, values = decoder(
                last,
                kv_cache,
                n_layer_cross_k, 
                n_layer_cross_v, 
                offset
            )
            kv_cache, offset = update_kv_cache(kv_cache, keys, values, offset)
            last = get_token(logits)
            tokens = torch.cat([tokens, last], dim=-1)
            
            # when to stop
            if last.item() == 50257:
                break
        
        decoded_text = tokenizer.decode(tokens[0])
    print("Inference completed successfully.")
    print("Decoded text:", decoded_text)
    print(f"The number of iterations by decoder: {iteration}")

    output_file_count = len(os.listdir(output_csv_file_path))
    
    # Convert to DataFrame
    csv_file_path = f"{output_csv_file_path}_infra_output_{output_file_count}_{datetime.datetime.now()}.csv"
    pd.DataFrame({
        "transcription": [decoded_text]
    }).to_csv(
        csv_file_path,
        index=False
    )
    print(f"Output saved to {csv_file_path}")
