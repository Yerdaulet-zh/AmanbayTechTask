# 🗣️ Whisper CLI Inference

This project provides a simple CLI and Jupyter Notebooks (analysis) to transcribe speech from an audio file using a modified and optimized Whisper v3 model.

Overall project directory:
<pre>
.
├── app
│   ├── assets
│   │   └── mel_filters.npz
│   ├── configs.py
│   ├── main.py
│   ├── main_utils.py
│   ├── model
│   │   └── medium.pt
│   ├── output
│   │   └── _infra_output_2_2025-07-02 11:21:41.097204.csv
│   ├── utils.py
│   └── whisper.py
├── LICENSE
├── notebooks
│   ├── audio_embeddings
│   │   ├── embeddings_0.npz
│   │   ├── ...
│   ├── audio_samples
│   │   └── english_sample.wav
│   ├── Fleurs.ipynb
│   ├── fleurs_metadata.csv
│   ├── Task3.ipynb
│   └── Разбор Архитектуры.ipynb
├── README.md
└── requirements.txt

</pre>


## ⚙️ Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt

    Note: Make sure ffmpeg is installed for audio processing if not already available.


▶️ How to Run

Run the inference script from the root of your project:

python3 app/main.py \
  --audio_file_path path/to/your_audio.wav \
  --max_token_sequence 50

Arguments

    --audio_file_path: (Required) Path to the input audio file (e.g., .wav, .flac).

    --max_token_sequence: (Optional) Maximum number of decoding steps (default: 50).
    Acceptable range is 1 to 100.

Example

python3 app/main.py \
  --audio_file_path audio_samples/english_sample.wav \
  --max_token_sequence 75

📤 Output

The transcribed text is saved as a .csv file in the output/ directory.
