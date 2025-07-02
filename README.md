# AmanbayTechTask
Testual task from AmanbayTech company aimed to validate the skills and depth of understanding

# 🗣️ Whisper CLI Inference

This project provides a simple CLI to transcribe speech from an audio file using a modified and optimized Whisper v3 model.

## 📁 Project Structure

.
├── app/
│ ├── main.py # Entry point for inference
│ ├── utils.py # Audio preprocessing utilities
│ ├── main_utils.py # Model loading and inference logic
│ ├── configs.py # Configuration (model path, device, etc.)
├── model/ # Folder where model will be downloaded (automatically created)
├── output/ # Folder where inference output will be saved (automatically created)


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
