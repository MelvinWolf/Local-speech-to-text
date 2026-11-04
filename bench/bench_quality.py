"""
Run each WAV in bench/recordings/ through the model with several chunk_duration_s
values and print transcriptions plus wall-clock transcribe time.

If <name>.txt exists next to <name>.wav, treat it as ground truth and print a
PASS/FAIL line based on exact match.
"""
import os
import sys
import time
import wave
import numpy as np
import onnx_asr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(ROOT, "bench", "recordings")
SAMPLE_RATE = 16000
CHUNK_DURATIONS = [0, 20, 30, 60]   # 0 = no chunking
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
PROVIDERS = ["DmlExecutionProvider"]


def load_wav(path):
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, f"{path} must be mono"
        assert wf.getframerate() == SAMPLE_RATE, f"{path} must be {SAMPLE_RATE} Hz"
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_with_chunking(model, audio, chunk_duration_s):
    if chunk_duration_s <= 0:
        result = model.recognize(audio, sample_rate=SAMPLE_RATE)
        return result.strip() if result else ""
    chunk_size = SAMPLE_RATE * chunk_duration_s
    texts = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue
        result = model.recognize(chunk, sample_rate=SAMPLE_RATE)
        if result and result.strip():
            texts.append(result.strip())
    return " ".join(texts)


def main():
    wavs = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav"))
    if not wavs:
        print(f"No WAVs in {RECORDINGS_DIR}. Run bench/record.py first.")
        sys.exit(1)

    print(f"Loading model {MODEL_NAME}...")
    model = onnx_asr.load_model(MODEL_NAME, providers=PROVIDERS)
    print("Model ready.\n")

    for wav in wavs:
        wav_path = os.path.join(RECORDINGS_DIR, wav)
        gt_path = wav_path[:-4] + ".txt"
        ground_truth = None
        if os.path.exists(gt_path):
            with open(gt_path, encoding="utf-8") as f:
                ground_truth = f.read().strip()

        audio = load_wav(wav_path)
        duration = len(audio) / SAMPLE_RATE
        print(f"=== {wav}  ({duration:.1f}s) ===")

        for cd in CHUNK_DURATIONS:
            t0 = time.perf_counter()
            try:
                text = transcribe_with_chunking(model, audio, cd)
                elapsed = time.perf_counter() - t0
                label = "no chunking" if cd == 0 else f"{cd}s chunks"
                status = ""
                if ground_truth is not None:
                    status = "  [PASS]" if text.strip() == ground_truth else "  [FAIL]"
                print(f"  [{label:12}]  {elapsed:5.2f}s{status}")
                print(f"    -> {text}")
            except Exception as e:
                print(f"  [{cd}s]  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
