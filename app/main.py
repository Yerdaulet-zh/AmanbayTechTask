import configs
import argparse
import soundfile as sf

from utils import compute_features

from main_utils import (
    load_tokenizer,
    ensure_directory,
    ensure_model,
    run_inference,
    load_model
)


def main(
    audio_file_path: str, 
    max_token_sequence: int
) -> None:
    ensure_directory(directory='model/')
    ensure_directory(directory='output/')

    ensure_model(
        model_path=configs.model_path, 
        medium_model_url=configs.medium_model_url
    )

    encoder, decoder = load_model(
        model_path=configs.model_path, 
        device=configs.device
    )

    tokenizer = load_tokenizer()

    print("Computing features from the audio file...")

    wave, sr = sf.read(
        audio_file_path, 
        dtype='float32'
    )

    mel = compute_features(
        wave=wave, 
        sample_rate=sr, 
        device=configs.device
    )

    run_inference(
        encoder=encoder,
        decoder=decoder, 
        mel=mel,
        tokenizer=tokenizer,
        device=configs.device,
        output_csv_file_path='output/',
        max_token_sequence=max_token_sequence
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CLI interface for running modified & optimised whisper v3 model")

    parser.add_argument("--audio_file_path", type=str, required=True, help="Path to the input audio file")
    # parser.add_argument("--output_csv_file_path", type=str, required=True, help="Path to the output .csv file")
    parser.add_argument("--max_token_sequence", type=int, default=50, help="Maximum number of iterations to generate tokens by decoder (default: 50)")

    args = parser.parse_args()

    main(args.audio_file_path, args.max_token_sequence)
