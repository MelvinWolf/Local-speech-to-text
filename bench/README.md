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
