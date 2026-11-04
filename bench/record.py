"""
Record a single WAV from the configured input device into bench/recordings/<name>.

Usage:
    python bench/record.py <name>.wav

Press Enter to start, press Enter again to stop.
"""
import json
import os
import sys
import threading
import wave
import numpy as np
import sounddevice as sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(ROOT, "bench", "recordings")
CONFIG_FILE = os.path.join(ROOT, "config.json")
SAMPLE_RATE = 16000


def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".wav"):
        print("Usage: python bench/record.py <name>.wav")
        sys.exit(2)

    out_path = os.path.join(RECORDINGS_DIR, sys.argv[1])
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    cfg = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    device_name = cfg.get("input_device")

    device_idx = None
    if device_name:
        for i, d in enumerate(sd.query_devices()):
            if d.get("name") == device_name and d.get("max_input_channels", 0) > 0:
                device_idx = i
                break

    chunks = []
    collecting = threading.Event()
    collecting.set()

    def cb(indata, frames, time_info, status):
        if collecting.is_set():
            chunks.append(indata.copy())

    print(f"Recording into {out_path}.")
    print("Press Enter to START.")
    input()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=cb, device=device_idx,
    )
    stream.start()
    print("Recording... press Enter to STOP.")
    input()
    collecting.clear()
    stream.stop()
    stream.close()

    audio = np.concatenate(chunks).flatten()
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved {len(audio) / SAMPLE_RATE:.2f}s to {out_path}")


if __name__ == "__main__":
    main()
