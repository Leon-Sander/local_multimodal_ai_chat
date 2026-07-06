from transformers import pipeline
import librosa
import io
from utils import load_config, timeit
import os
import subprocess
config = load_config()

def convert_webm_to_wav_ffmpeg(audio_bytes):
    # Save the WebM bytes to a file
    with open("temp_audio.webm", "wb") as f:
        f.write(audio_bytes)

    # Use FFmpeg to convert WebM to WAV
    result = subprocess.run(
        ["ffmpeg", "-fflags", "+igndts", "-i", "temp_audio.webm", "-c:a", "pcm_s16le", "temp_audio.wav"],
        capture_output=True
    )

    if result.returncode != 0:
        print(result.stderr.decode())
        raise RuntimeError("FFmpeg failed to convert WebM to WAV")

    # Read the WAV file back into memory
    with open("temp_audio.wav", "rb") as f:
        wav_data = f.read()

    wav_io = io.BytesIO(wav_data)

    # Clean up the temp files
    os.remove("temp_audio.webm")
    os.remove("temp_audio.wav")

    return wav_io

def convert_bytes_to_array(audio_bytes):
    try:
        audio_bytes_io = io.BytesIO(audio_bytes)
        audio, sample_rate = librosa.load(audio_bytes)
    except Exception as e:
        print("Audio error, trying to convert to wav.")
        wav_io = convert_webm_to_wav_ffmpeg(audio_bytes)
        audio, sample_rate = librosa.load(wav_io)

    print(sample_rate)
    return audio

@timeit
def transcribe_audio(audio_bytes):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=config["whisper_model"],
        chunk_length_s=30,
        device=device,
    )   

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]

    return prediction
