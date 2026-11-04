# Voice Input Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three reliability issues in the voice input tool: wrong audio device leaking in, end-of-utterance loss when stopping the recording promptly, and broken text caused by hard chunking at 20 s boundaries.

**Architecture:** Surgical edits to `voice_input.py` and `setup.py`. Two new optional config keys (`input_device`, `chunk_duration_s`) with safe defaults. A new `bench/` harness validates the behavior and lets us tune empirically — there is no existing test framework in this project, so behavioral verification is integration-style via the bench, not unit tests. The one helper that has pure logic (`_resolve_input_device`) is small enough to verify by reading.

**Tech Stack:** Python 3.10+, sounddevice (PortAudio), onnx-asr, numpy, Windows ctypes (already used). No new runtime dependencies. The bench scripts use only existing dependencies plus stdlib `wave` for WAV I/O.

---

## Spec reference

See [docs/superpowers/specs/2026-05-06-voiceinput-fixes-design.md](../specs/2026-05-06-voiceinput-fixes-design.md) for goals, non-goals, and verification criteria.

## Parallel execution map

Three tracks may be implemented concurrently by separate subagents because each touches a different file:

| Track | File(s) | Tasks |
|---|---|---|
| A — voice_input.py | `voice_input.py` | 1, 2, 3 (sequential within the file) |
| B — setup.py | `setup.py` | 4, 5 (sequential within the file) |
| C — bench harness | `bench/*` (new) | 6, 7, 8 |

Task 9 (manual verification) is sequential and runs after all three tracks land.

## Preconditions

- The repo is currently NOT under git (`git status` returns "not a repository"). The plan assumes the user runs `git init` before starting; if they choose not to, the "commit" steps are no-ops and can be skipped without affecting correctness.
- `python voice_input.py` currently runs on Windows with the configured hotkey. After each task that modifies the running app, restart `voice_input.py` to pick up changes.
- Throughout, sample rate is fixed at 16000 Hz; do not parameterize.

## File structure after implementation

```
voice_input.py                          (modified)
setup.py                                (modified)
config.json                             (no schema change required for existing users — new keys optional)
bench/                                  (new, tracked)
  README.md                             (instructions for use)
  record.py                             (capture a single WAV into recordings/)
  bench_quality.py                      (offline, runs WAVs against the model with varying chunk_duration_s)
  bench_latency.py                      (interactive, measures stop-drain strategies)
  recordings/                           (gitignored)
    .gitkeep
.gitignore                              (new)
docs/superpowers/specs/...              (already written)
docs/superpowers/plans/...              (this file)
```

---

## Task 1: Wire pinned input device into stream open (Track A)

**Files:**
- Modify: [voice_input.py](../../voice_input.py) — add helper, read config key, pass `device=` to `sd.InputStream`, log the opened device.

- [ ] **Step 1: Add the device-resolution helper above the `# --- Audio ---` section in [voice_input.py:98](../../voice_input.py#L98)**

Insert immediately after the model-loading block (which ends at line 96):

```python
# --- Device resolution ---
def _resolve_input_device(name, devices):
    """
    Return the index of the first input-capable device whose name matches `name`.
    Returns None if name is falsy or no match is found.
    `devices` is the iterable returned by sd.query_devices().
    """
    if not name:
        return None
    for i, dev in enumerate(devices):
        if dev.get("name") == name and dev.get("max_input_channels", 0) > 0:
            return i
    return None
```

- [ ] **Step 2: Read the new config key alongside the others**

Modify the config-reading block at [voice_input.py:34-41](../../voice_input.py#L34-L41) to add one line:

```python
MODE             = cfg.get("mode", "toggle")
HOTKEY_VK        = cfg.get("hotkey_vk", 0x6B)
HOTKEY_NAME      = cfg.get("hotkey_name", "Numpad +")
VAD_SILENCE_MS   = cfg.get("vad_silence_ms", 500)
MODEL_NAME       = cfg.get("model", "nemo-parakeet-tdt-0.6b-v3")
PROVIDERS        = cfg.get("providers", ["DmlExecutionProvider"])
INPUT_DEVICE     = cfg.get("input_device")          # NEW: human-readable device name, optional
SAMPLE_RATE      = 16000
CHUNK_DURATION   = 20
```

- [ ] **Step 3: Resolve the device and pass it to `sd.InputStream` at [voice_input.py:106-109](../../voice_input.py#L106-L109)**

Replace:

```python
audio_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    callback=audio_callback
)
audio_stream.start()
log.info("Audio stream open.")
```

with:

```python
_devices = sd.query_devices()
_device_idx = _resolve_input_device(INPUT_DEVICE, _devices)
if INPUT_DEVICE and _device_idx is None:
    log.warning(
        f"Configured input_device {INPUT_DEVICE!r} not found among current devices; "
        f"falling back to system default."
    )
audio_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    callback=audio_callback,
    device=_device_idx,
)
audio_stream.start()
_opened = sd.query_devices(_device_idx if _device_idx is not None else sd.default.device[0])
log.info(f"Audio stream open on device: {_opened['name']!r} (hostapi {_opened['hostapi']})")
```

- [ ] **Step 4: Manually verify**

Run:
```bash
python c:/Users/melvi/Voiceinput/voice_input.py
```

Then in a second shell:
```bash
tail -n 5 c:/Users/melvi/Voiceinput/voice_input.log
```

Expected: log shows `Audio stream open on device: '<name>' (hostapi <int>)`. The name should match your current default microphone (since `input_device` is still unset in `config.json`).

Stop voice_input.py (Task Manager → end pythonw.exe / python.exe) before continuing.

- [ ] **Step 5: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add voice_input.py && git commit -m "feat(voice_input): pin input device by name from config"
```

(Skip if not under git.)

---

## Task 2: Add chunking-disabled branch to `transcribe_chunks` (Track A)

**Files:**
- Modify: [voice_input.py:114-124](../../voice_input.py#L114-L124) — branch on a new config knob.

- [ ] **Step 1: Add the new config key**

In the config-reading block (modified in Task 1), add one line:

```python
CHUNK_DURATION_S = cfg.get("chunk_duration_s", 0)   # NEW: 0 = no chunking, feed whole audio in one call
```

Place it directly below the `INPUT_DEVICE` line you added.

- [ ] **Step 2: Branch on `CHUNK_DURATION_S` in `transcribe_chunks`**

Replace [voice_input.py:114-124](../../voice_input.py#L114-L124) with:

```python
def transcribe_chunks(audio):
    if CHUNK_DURATION_S <= 0:
        result = model.recognize(audio, sample_rate=SAMPLE_RATE)
        return result.strip() if result else ""
    chunk_size = SAMPLE_RATE * CHUNK_DURATION_S
    texts = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue
        result = model.recognize(chunk, sample_rate=SAMPLE_RATE)
        if result and result.strip():
            texts.append(result.strip())
    return " ".join(texts)
```

Notes:
- The legacy module-level constant `CHUNK_DURATION = 20` at [voice_input.py:41](../../voice_input.py#L41) is now unused. Remove it in the same edit.
- The `n_chunks` log line at [voice_input.py:137](../../voice_input.py#L137) currently divides by `CHUNK_DURATION`. Update it to use `CHUNK_DURATION_S` and skip the division when chunking is disabled:

Replace [voice_input.py:137-138](../../voice_input.py#L137-L138):

```python
n_chunks = int(np.ceil(duration / CHUNK_DURATION))
log.info(f"Transcribing ({n_chunks} chunk(s))...")
```

with:

```python
if CHUNK_DURATION_S > 0:
    n_chunks = int(np.ceil(duration / CHUNK_DURATION_S))
    log.info(f"Transcribing ({n_chunks} chunk(s))...")
else:
    log.info("Transcribing (single call)...")
```

- [ ] **Step 3: Manually verify with a short utterance**

Run:
```bash
python c:/Users/melvi/Voiceinput/voice_input.py
```

Press hotkey, speak "this is a test", press hotkey again.

Check log:
```bash
tail -n 10 c:/Users/melvi/Voiceinput/voice_input.log
```

Expected: log line `Transcribing (single call)...` and a sensible transcription.

Stop voice_input.py.

- [ ] **Step 4: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add voice_input.py && git commit -m "feat(voice_input): make chunk duration configurable, default to no chunking"
```

---

## Task 3: Add stop-drain to `toggle()` (Track A)

**Files:**
- Modify: [voice_input.py:155-166](../../voice_input.py#L155-L166) — change the toggle handler so the audio stream is drained before processing and restarted afterwards.

- [ ] **Step 1: Replace `toggle()` body**

Replace [voice_input.py:155-166](../../voice_input.py#L155-L166):

```python
def toggle():
    global collecting, audio_chunks
    try:
        if not collecting:
            audio_chunks = []
            collecting = True
            log.info("Recording...")
        else:
            collecting = False
            threading.Thread(target=transcribe_and_paste, daemon=True).start()
    except Exception as e:
        log.error(f"toggle error: {e}")
```

with:

```python
def toggle():
    global collecting, audio_chunks
    try:
        if not collecting:
            audio_chunks = []
            if not audio_stream.active:
                audio_stream.start()
            collecting = True
            log.info("Recording...")
        else:
            # Stop FIRST so PortAudio drains pending buffers into the callback
            # while the gate is still open. Closing the gate first would cause
            # those flushed callbacks to drop their audio.
            audio_stream.stop()
            collecting = False
            snapshot = list(audio_chunks)
            # Resume the stream now so the next start hotkey is instant.
            audio_stream.start()
            threading.Thread(
                target=transcribe_and_paste, args=(snapshot,), daemon=True
            ).start()
    except Exception as e:
        log.error(f"toggle error: {e}")
```

Why a snapshot: the next recording reuses the global `audio_chunks` (re-initialized to `[]` on the next start). Passing a list copy to the transcription thread isolates it from any future mutations.

Why `stop()` precedes `collecting = False`: the audio callback gates appends on `collecting`. PortAudio's `stream.stop()` blocks until pending input buffers have been delivered to the callback. If the gate were closed before `stop()`, those flushed callbacks would see `collecting = False` and drop their audio — defeating the drain.

- [ ] **Step 2: Manually verify the trailing-word fix**

Run:
```bash
python c:/Users/melvi/Voiceinput/voice_input.py
```

Press hotkey, speak each of these phrases naturally and press the hotkey *immediately* after the last word:

- "check this now"
- "send the code"
- "that looks right"

Inspect the log. The trailing word ("now", "code", "right") should appear verbatim in the transcription. Before this change, observation was that "now" became "nothing" and similar mutations happened on long utterances.

Stop voice_input.py.

- [ ] **Step 3: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add voice_input.py && git commit -m "fix(voice_input): drain audio stream on stop to capture full utterance"
```

---

## Task 4: Add Microphone selection step to setup wizard (Track B)

**Files:**
- Modify: [setup.py](../../setup.py) — add a wizard step between the hotkey step and the model step.

- [ ] **Step 1: Add input-device enumeration helper**

At the top of [setup.py](../../setup.py), in the imports block (after line 10), add:

```python
import sounddevice as sd
```

- [ ] **Step 2: Insert the Microphone wizard step in `main()`**

In `setup.py`, locate the model selection at [setup.py:167-169](../../setup.py#L167-L169):

```python
    # --- Model ---
    model_idx = menu("Model:", [m["label"] for m in MODELS])
    config["model"] = MODELS[model_idx]["key"]
```

Insert *immediately above* it:

```python
    # --- Microphone ---
    inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    options = ["(use Windows default input device)"] + [
        f"{d['name']}  [hostapi {d['hostapi']}, channels {d['max_input_channels']}]"
        for d in inputs
    ]
    mic_idx = menu("Microphone:", options)
    if mic_idx == 0:
        config.pop("input_device", None)
    else:
        config["input_device"] = inputs[mic_idx - 1]["name"]
```

- [ ] **Step 3: Update the summary print at the bottom of `main()`**

Append after [setup.py:187](../../setup.py#L187) (`print(f"  Provider : {config['providers'][0]}")`), before the closing line:

```python
    if "input_device" in config:
        print(f"  Mic      : {config['input_device']}")
    else:
        print(f"  Mic      : (system default)")
```

- [ ] **Step 4: Manually verify**

Run:
```bash
python c:/Users/melvi/Voiceinput/setup.py
```

Walk through the wizard. The Microphone step should list at least the default mic and "(use Windows default input device)". Choose one, finish the wizard, and confirm `config.json` either contains `input_device: "<name>"` or omits the key:

```bash
cat c:/Users/melvi/Voiceinput/config.json
```

- [ ] **Step 5: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add setup.py && git commit -m "feat(setup): add microphone selection step"
```

---

## Task 5: Add Chunk Duration step to setup wizard (Track B)

**Files:**
- Modify: [setup.py](../../setup.py) — add a small input prompt for `chunk_duration_s`.

- [ ] **Step 1: Add a `prompt_int_with_default` helper near the top of `setup.py`**

Place it directly below the `menu` function defined at [setup.py:92](../../setup.py#L92):

```python
def prompt_int_with_default(prompt, default):
    raw = input(f"{prompt}  [default {default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
```

- [ ] **Step 2: Insert the Chunk Duration step**

Place it directly below the Microphone step you added in Task 4 and above the Model step:

```python
    # --- Chunk duration ---
    print("\nChunk duration: how long (seconds) before audio is split for transcription.")
    print("  0 = no splitting (recommended for cleaner sentences)")
    print("  20 = old default (only set if long recordings cause issues on your hardware)")
    chunk_s = prompt_int_with_default("Chunk duration (s)", existing.get("chunk_duration_s", 0))
    config["chunk_duration_s"] = max(0, chunk_s)
```

- [ ] **Step 3: Update the summary print**

Below the `Mic` summary line you added in Task 4, append:

```python
    print(f"  Chunking : {config['chunk_duration_s']}s" if config['chunk_duration_s'] > 0 else "  Chunking : disabled")
```

- [ ] **Step 4: Manually verify**

Run:
```bash
python c:/Users/melvi/Voiceinput/setup.py
```

Press Enter at the chunk duration prompt to accept the default, finish the wizard, then:

```bash
cat c:/Users/melvi/Voiceinput/config.json
```

Expected: `chunk_duration_s: 0` is in the file.

- [ ] **Step 5: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add setup.py && git commit -m "feat(setup): add chunk duration prompt"
```

---

## Task 6: Create bench/ scaffolding (Track C)

**Files:**
- Create: `bench/README.md`, `bench/record.py`, `bench/recordings/.gitkeep`
- Create: `.gitignore` (new at repo root)

- [ ] **Step 1: Create `.gitignore` at repo root**

Create [.gitignore](../../../.gitignore):

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Local artifacts
voice_input.log
config.json

# Bench reference recordings (user voice, do not commit)
bench/recordings/*
!bench/recordings/.gitkeep
```

`config.json` is gitignored because it is generated by `setup.py` and contains user-specific paths/devices.

- [ ] **Step 2: Create `bench/recordings/.gitkeep`**

Create the file `bench/recordings/.gitkeep` with empty content. This preserves the directory in git.

- [ ] **Step 3: Create `bench/README.md`**

```markdown
# Voice Input Bench Harness

Two scripts for verifying behavior changes empirically. Reference recordings are stored under `bench/recordings/` and are gitignored — they hold your voice and should not be shared.

## One-time: record reference samples

Record at least these four samples in normal use conditions (your usual mic, room, distance):

```
python bench/record.py short.wav   # ~5 seconds, one sentence
python bench/record.py medium.wav  # ~25 seconds, paragraph that crosses the 20s mark
python bench/record.py long.wav    # ~60 seconds
python bench/record.py xlong.wav   # ~120 seconds, only used to probe the upper bound
```

Press Enter to start each recording, press Enter again to stop. Files are saved to `bench/recordings/`.

Optional: create a hand-corrected `bench/recordings/<name>.txt` next to each WAV with the ground-truth transcript. `bench_quality.py` will diff against it if present.

## Issue 3 — chunking quality

```
python bench/bench_quality.py
```

Runs each WAV under `bench/recordings/` through the model with `chunk_duration_s` set to 0, 20, 30, and 60. Prints the transcription and the wall-clock transcribe time for each. If a `.txt` ground-truth file exists, prints a per-WAV pass/fail line based on whether transcripts match exactly (eyeball the diff in the printout to interpret partial matches).

## Issue 2 — stop-drain latency and trailing-word integrity

```
python bench/bench_latency.py --strategy=drain
python bench/bench_latency.py --strategy=baseline
python bench/bench_latency.py --strategy=tail --tail-ms=200
```

Interactive. Prompts you to speak a known phrase (e.g., "check this now"), captures from the configured input device, applies the chosen strategy at "stop", transcribes, and prints both the wall-clock latency from stop → result and the transcript. Run each strategy 10×, eyeball the trailing word in the transcripts, then compare strategies.

CSV output is written to `bench/results-<strategy>-<timestamp>.csv` for offline comparison.
```

- [ ] **Step 4: Create `bench/record.py`**

```python
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
```

- [ ] **Step 5: Manually verify scaffolding**

Run:
```bash
python c:/Users/melvi/Voiceinput/bench/record.py short.wav
```

Press Enter, speak "this is a test", press Enter. Confirm:

```bash
ls c:/Users/melvi/Voiceinput/bench/recordings/
```

Expected: `short.wav` is present, ~3-5 seconds long, plays back as your voice.

- [ ] **Step 6: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add .gitignore bench/ && git commit -m "feat(bench): add bench harness scaffolding and record.py"
```

---

## Task 7: Implement `bench/bench_quality.py` (Track C)

**Files:**
- Create: `bench/bench_quality.py`

- [ ] **Step 1: Create the file**

```python
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
```

- [ ] **Step 2: Manually verify**

After Task 6 produced at least `short.wav`, run:

```bash
python c:/Users/melvi/Voiceinput/bench/bench_quality.py
```

Expected: prints one block per WAV with four sub-rows (`no chunking`, `20s chunks`, `30s chunks`, `60s chunks`), each with a transcribe time and the transcribed text. The first run loads the model (~10–20 s), subsequent runs are faster.

For a short utterance, expect all four rows to produce identical or near-identical text. The interesting cases come from medium/long.wav.

- [ ] **Step 3: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add bench/bench_quality.py && git commit -m "feat(bench): add quality benchmark across chunk durations"
```

---

## Task 8: Implement `bench/bench_latency.py` (Track C)

**Files:**
- Create: `bench/bench_latency.py`

- [ ] **Step 1: Create the file**

```python
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
```

- [ ] **Step 2: Manually verify a single take**

Run with `--takes=1` for a quick smoke check:

```bash
python c:/Users/melvi/Voiceinput/bench/bench_latency.py --strategy=drain --takes=1
```

Press Enter through the prompts and speak the first phrase. Expected: a printed line with `<latency> ms   '<transcript>'` and a CSV file under `bench/`. The trailing word ("now") should appear verbatim.

- [ ] **Step 3: Commit**

```bash
cd c:/Users/melvi/Voiceinput && git add bench/bench_latency.py && git commit -m "feat(bench): add latency + trailing-word benchmark for stop strategies"
```

---

## Task 9: Manual verification & decision checkpoint

This task is for the user (and for the executing agent to drive the user through). It is the final gate before considering the work done.

- [ ] **Step 1: Record reference WAVs (one-time)**

```bash
python c:/Users/melvi/Voiceinput/bench/record.py short.wav    # ~5s
python c:/Users/melvi/Voiceinput/bench/record.py medium.wav   # ~25s
python c:/Users/melvi/Voiceinput/bench/record.py long.wav     # ~60s
python c:/Users/melvi/Voiceinput/bench/record.py xlong.wav    # ~120s
```

Optionally create a `bench/recordings/<name>.txt` with the exact transcript next to each WAV.

- [ ] **Step 2: Run quality benchmark to validate issue 3**

```bash
python c:/Users/melvi/Voiceinput/bench/bench_quality.py
```

**Decision criterion:**
- If `no chunking` produces clean transcripts on `medium.wav` and `long.wav` and runs in under 2× the chunked-baseline time, keep `chunk_duration_s = 0` as the default. **Done.**
- If `no chunking` errors out (OOM, exception) on `xlong.wav`, note the longest size that worked. Set the default to ~80% of that ceiling. Update `setup.py` Step 2 default and `voice_input.py` Step 2 default to match. Re-commit.
- If `no chunking` is materially slower (> 2×) at long lengths but still produces correct text, keep `0` as default but document the tradeoff in `bench/README.md`.

- [ ] **Step 3: Run latency benchmark to validate issue 2**

Run all three strategies, 10 takes each:

```bash
python c:/Users/melvi/Voiceinput/bench/bench_latency.py --strategy=baseline
python c:/Users/melvi/Voiceinput/bench/bench_latency.py --strategy=drain
python c:/Users/melvi/Voiceinput/bench/bench_latency.py --strategy=tail --tail-ms=200
```

Open the three CSVs side by side. **Pass criteria for `drain`:**
- p95 latency is within 50 ms of baseline.
- Trailing word is correct in ≥ 95% of takes (i.e., 0 or 1 wrong out of 50 across the five phrases × ten takes).

If `drain` fails the trailing-word criterion, fall back to **Approach C** (always-collecting ring buffer):
- Move the `chunks` accumulator out of the toggle gate in `voice_input.py` Task 3 so the callback always appends to a deque-style ring buffer with a bounded size (say 60 s @ 16 kHz = 960k samples).
- On stop, slice using callback timestamps: track `t_start` and `t_stop` via `time.monotonic()` set in `toggle()`, and discard frames outside that window using `time_info.inputBufferAdcTime`.
- This is a follow-up — do not implement it preemptively. Only if drain fails on the user's hardware.

- [ ] **Step 4: Validate issue 1 (device pinning)**

```bash
python c:/Users/melvi/Voiceinput/setup.py     # pick a specific microphone (not "default")
python c:/Users/melvi/Voiceinput/voice_input.py
tail -n 5 c:/Users/melvi/Voiceinput/voice_input.log
```

Expected: log shows `Audio stream open on device: '<chosen device name>' (hostapi <int>)`. Speak something to confirm transcription still works on the pinned device.

To verify the leak fix: if your system has a "Stereo Mix" or virtual loopback input, change Windows default *Input* to that loopback, leave `voice_input.py` running, and confirm via the log that voice_input is still on your real microphone.

- [ ] **Step 5: Final commit / cleanup**

If any tuning decisions resulted in further code changes (default for `chunk_duration_s`, fallback to Approach C, etc.):

```bash
cd c:/Users/melvi/Voiceinput && git add -u && git commit -m "tune: voice_input defaults from bench results"
```

If everything passed without changes, no commit is needed.

---

## Self-review

**Spec coverage:**

| Spec section | Plan coverage |
|---|---|
| Issue 1 — pinned input device | Tasks 1, 4 |
| Issue 2 — stop without losing tail (Approach A) | Task 3 |
| Issue 2 — fallback to Approach C | Task 9 Step 3 (conditional, only on bench failure) |
| Issue 3 — configurable chunking | Tasks 2, 5 |
| Bench harness scaffolding | Tasks 6, 7, 8 |
| Verification criteria | Task 9 |

All spec sections have explicit tasks. ✓

**Placeholder scan:** no TBD/TODO/"as appropriate" content. Code blocks are complete in every step. ✓

**Type / name consistency:**
- `_resolve_input_device(name, devices)` — defined in Task 1, used in Task 1 only. ✓
- `INPUT_DEVICE` config var — set in Task 1 step 2, used in Task 1 step 3. ✓
- `CHUNK_DURATION_S` — set in Task 2 step 1, used in Task 2 step 2. The legacy `CHUNK_DURATION` (no `_S`) is explicitly removed in Task 2 step 2. ✓
- `prompt_int_with_default` — defined and used in Task 5. ✓
- `chunk_duration_s` config key — written in Task 5, read in Task 2, surfaced in `bench_quality.py` (Task 7) by name (string match). ✓

**Scope check:** three issues + bench, all in two existing files plus one new directory. Single plan is appropriate. ✓

