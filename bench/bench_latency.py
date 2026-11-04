"""
Interactive harness for issue 2 (end-of-utterance loss).

For a chosen "stop" strategy, prompt the user to speak a phrase, capture audio,
apply the strategy at stop time, transcribe, and report:
  - wall-clock latency from "stop" event to transcript availability
  - the transcript itself (eyeball trailing word integrity)

Strategies:
  --strategy=baseline           flip a flag and process whatever was buffered (today's behavior)
  --strategy=drain              call stream.stop() before processing (Approach A)
  --strategy=tail --tail-ms=N   sleep N ms before processing (Approach B for comparison)

Records to bench/results-<strategy>-<timestamp>.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import threading
import time
import numpy as np
import sounddevice as sd
import onnx_asr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT, "config.json")
RESULTS_DIR = os.path.join(ROOT, "bench")
SAMPLE_RATE = 16000

PHRASES = [
    "check this now",
    "send the code",
    "that looks right",
    "I will be there soon",
    "let me know what you think",
]
TAKES_PER_PHRASE = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["baseline", "drain", "tail"], required=True)
    p.add_argument("--tail-ms", type=int, default=200)
    p.add_argument("--takes", type=int, default=TAKES_PER_PHRASE)
    return p.parse_args()


def load_device_index():
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    name = cfg.get("input_device")
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d.get("name") == name and d.get("max_input_channels", 0) > 0:
            return i
    return None


def main():
    args = parse_args()
    device_idx = load_device_index()

    print(f"Loading model...")
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=["DmlExecutionProvider"])
    print("Model ready.\n")

    chunks = []
    collecting = threading.Event()

    def cb(indata, frames, time_info, status):
        if collecting.is_set():
            chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=cb, device=device_idx,
    )
    stream.start()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"results-{args.strategy}-{timestamp}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["phrase", "take", "strategy", "tail_ms", "stop_to_result_ms", "transcript"])
        for phrase in PHRASES:
            print(f"\n--- Phrase: {phrase!r} ({args.takes} takes) ---")
            for i in range(args.takes):
                input(f"  Take {i+1}: press Enter, then speak {phrase!r}, then press Enter to stop. ")
                chunks.clear()
                collecting.set()
                input("    [recording... Enter to stop]")
                t_stop = time.perf_counter()
                # apply strategy
                if args.strategy == "baseline":
                    collecting.clear()
                elif args.strategy == "drain":
                    stream.stop()        # drain pending buffers with gate still open
                    collecting.clear()
                    stream.start()
                elif args.strategy == "tail":
                    time.sleep(args.tail_ms / 1000.0)
                    collecting.clear()

                snap = list(chunks)
                if not snap:
                    print("    (no audio captured)")
                    continue
                audio = np.concatenate(snap).flatten()
                result = model.recognize(audio, sample_rate=SAMPLE_RATE)
                t_done = time.perf_counter()
                latency_ms = (t_done - t_stop) * 1000.0
                transcript = (result or "").strip()
                print(f"    {latency_ms:6.1f} ms   {transcript!r}")
                writer.writerow([phrase, i + 1, args.strategy, args.tail_ms, f"{latency_ms:.1f}", transcript])
                fp.flush()

    stream.stop()
    stream.close()
    print(f"\nResults: {out_csv}")


if __name__ == "__main__":
    main()
