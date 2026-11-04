# Voice Input — Reliability Fixes (2026-05-06)

## Goals

Address three recurring quality issues observed during daily use of `voice_input.py`:

1. The wrong audio source is occasionally captured (e.g., audio from a YouTube video instead of the microphone).
2. The end of an utterance is dropped when the hotkey is pressed promptly after speaking, especially on longer dictations. This manifests both as missing trailing words and as mutated trailing words (e.g., "now" transcribed as "nothing").
3. Hard chunking at 20 s produces broken sentence structure and duplicate words when the boundary lands mid-word.

Out of scope for this iteration: code-switching (EN main + DE words inside) and post-hoc text polish via a local LLM. Both are deferred.

## Non-goals

- No rewrite of the audio or hotkey stack. Changes stay within `voice_input.py`, `setup.py`, and `config.json`.
- No new runtime dependencies.
- No change to the toggle/handsfree mode contract or the hotkey API.
- No reduction in baseline latency below what the current code already achieves; the bar is "no worse, ideally better."

## Summary of changes

| Concern | Change |
|---|---|
| Wrong device | Persist a chosen input device by name in `config.json`; resolve to index at startup; log opened device. |
| End-of-utterance loss | Drain the audio stream cleanly on stop, falling back to a tunable tail if drain proves unreliable. |
| Hard 20 s chunking | Make `chunk_duration_s` configurable; default to `0` (no chunking); leave the existing chunked path as a fallback. |
| Verification | Add a `bench/` harness with reference recordings (gitignored) to measure latency and end-of-utterance integrity, plus long-form quality at varying chunk durations. |

## Issue 1 — Pinned input device

**Why:** `sd.InputStream` opens with no `device=` argument ([voice_input.py:106-109](../../voice_input.py#L106-L109)), so it tracks the Windows default input. The default can be changed by other applications, by Bluetooth headset connect/disconnect events, or by a virtual loopback device ("Stereo Mix") becoming the system default — at which point system audio (e.g., a YouTube video) is captured instead of the user's microphone.

**Design:**

- New `config.json` key: `input_device` (string, optional). When present, holds the human-readable device name. When absent, behavior is unchanged (default device).
- `setup.py` adds a "Microphone" wizard step:
  - Enumerate input devices via `sd.query_devices()` filtered to those with `max_input_channels > 0`.
  - Show numbered menu with the device name and hostapi (so duplicates from different APIs are distinguishable).
  - Save the selected device's `name` field to `config.json`.
  - Offer a "(use system default)" option that omits the key.
- `voice_input.py` startup:
  - If `input_device` is configured, scan `sd.query_devices()` for a matching `name`. If found, pass the index to `sd.InputStream(device=...)`. If not found, log a warning and fall back to the default device.
  - At open time, log the actual device name and hostapi being used. This makes `voice_input.log` self-diagnosing for "wrong device" reports.

**Trade-off:** if the user later renames a device or swaps hardware, they re-run `setup.py`. This is a deliberate trade — explicitness beats silent fallback.

## Issue 2 — Stop without losing the tail

**Why:** Pressing the hotkey flips `collecting=False` immediately ([voice_input.py:155-164](../../voice_input.py#L155-L164)) and spawns the transcription thread. Any audio still in the OS / sounddevice ring buffer never reaches `audio_callback` with `collecting=True`, so it is silently dropped. Because the model returns its best guess for whatever it does receive, partial-word truncations get rendered as the wrong full word rather than as missing words.

**Design — drain-first, fall back to tail buffer:**

The implementation goes in starting from approach A (cleanest, near-zero added latency). If the benchmark shows it loses leading frames after the stream restart, we move to approach C without changing the public surface.

- **Approach A (default): drain via `stream.stop()`.**
  On the stop hotkey:
  1. Call `audio_stream.stop()` first. PortAudio drains pending input buffers into the callback before this returns; the gate must still be open at this point so the callback appends those flushed buffers.
  2. Set `collecting = False`.
  3. Hand `audio_chunks` to the transcription thread.
  4. Restart the stream so the next start hotkey is instant.

  The order matters: closing the gate before `stop()` would cause the flushed callbacks to drop their audio, defeating the drain.

  Added latency: bounded by the depth of the audio buffer, typically ≤ 30 ms on WASAPI low-latency.

- **Approach C (fallback): always-collecting ring buffer + drain.**
  If the start/stop cycle drops the first frames after restart on the user's hardware, we keep the stream running continuously and gate inclusion by timestamps:
  1. The callback always appends to a fixed-size ring buffer (a few seconds of capacity is enough; we only ever read the most recent recording window).
  2. On start, record `t_start = time.monotonic()`.
  3. On stop, record `t_stop = time.monotonic()`, then call `audio_stream.stop()` to flush.
  4. Slice the buffer for samples in `[t_start, t_stop]` using callback timestamps.

  Added latency: same as A. The ring buffer simply removes the start/stop hardware dependency.

- **Approach B (rejected, kept for comparison): fixed-tail sleep.**
  Sleep `tail_ms` after stop before processing. Predictable but pure latency overhead, and the right `tail_ms` depends on the audio backend. Used in the benchmark only to validate that A actually captures the same audio at lower latency.

**Configurable knob:** none initially. If real-world use shows we need it, we add `stop_drain_strategy` later. YAGNI for now.

## Issue 3 — Configurable chunking

**Why:** `transcribe_chunks` ([voice_input.py:114-124](../../voice_input.py#L114-L124)) cuts audio at hard 20 s sample boundaries with no overlap and no silence-awareness. When the cut lands mid-word the model invents fragments at both edges, producing duplicate words and broken sentence structure.

**Design:**

- New `config.json` key: `chunk_duration_s` (number, default `0`). `0` means "no chunking — feed the whole audio to the model in a single call." A positive integer behaves as today.
- In `voice_input.py`:
  - When `chunk_duration_s == 0`, call `model.recognize(audio, sample_rate=SAMPLE_RATE)` exactly once on the full waveform. The existing minimum-length and post-processing logic stay intact.
  - When `chunk_duration_s > 0`, the existing chunked path runs unchanged.
- Default in newly-generated configs: `0`.

**Risk:** Parakeet TDT v3 may have a practical upper bound on input length due to ONNX export constraints or VRAM. The benchmark covers this by including 60 s, 90 s, and 120 s recordings; if any duration fails or regresses, we document the empirical safe ceiling and either keep `0` as default or drop the default down to the safe ceiling.

**Why not silence-aware splitting now:** if `0` works for typical dictation lengths (the user's longest recent recordings are ~50 s and ~74 s per the log), unbounded is strictly better — no boundary errors, no overlap merge logic. Silence-aware chunking is a fallback if unbounded doesn't hold up at longer durations.

## Benchmark harness

A new `bench/` directory, gitignored, with two scripts and a reference-recordings folder.

```
bench/
  recordings/             # gitignored — user's reference WAVs
  bench_latency.py        # interactive: measures stop-hotkey-to-paste latency
  bench_quality.py        # offline: runs WAVs through the model with varying settings
```

### `bench_latency.py`

Interactive. Loops a prompt: "Speak the test phrase, then press the hotkey to stop." For each strategy under test (A, B with various `tail_ms`, C), it:

- Times from hotkey-stop callback fired → text appearing on the clipboard.
- Records the audio waveform delivered to the model and saves it for inspection.
- Prints p50 / p95 latency and a flag if the trailing word of the test phrase appears verbatim, mutated, or missing in the transcription.

The user supplies test phrases ending in distinct trailing words: `"check this now"`, `"send the code"`, `"that looks right"`, plus one or two German trailing words. Each phrase is repeated 10 times per strategy.

### `bench_quality.py`

Offline. Reads WAVs from `bench/recordings/`, runs each through `model.recognize` with `chunk_duration_s` set to `[0, 20, 30, 60]`, prints per-WAV transcription, total transcribe time, and a side-by-side diff against an optional `<wavname>.txt` ground-truth file. The user does the qualitative read; we don't compute WER automatically (small N, mixed-language, not worth a tokenizer).

### Reference recordings

Recorded once by the user in normal conditions:

- 1 short utterance (~5 s, 1 sentence)
- 1 medium (~25 s, paragraph that crosses today's chunk boundary)
- 1 long (~60 s)
- 1 very long (~120 s, primarily to probe the upper bound)

These live in `bench/recordings/` and are not committed. A `bench/recordings/.gitkeep` and `bench/README.md` describe how to record them.

## Verification criteria

Before merging, all four must hold against the reference recordings:

- **Issue 1**: setup wizard offers device selection; opened device is named in `voice_input.log`; configuring a non-default device pipes that device's audio (verified by speaking into one mic with another set as system default).
- **Issue 2**: under approach A, p95 added latency vs. current behavior is within 50 ms, and trailing-word integrity is ≥ 95% across the 4 trailing-word phrases × 10 takes (i.e., the missing/mutated rate must drop dramatically vs. baseline).
- **Issue 3**: with `chunk_duration_s = 0`, the medium and long reference WAVs transcribe with no observable cross-boundary errors and total transcribe time is within 2× the chunked-baseline.
- **Regression check**: the short-utterance latency p50 is no worse than baseline.

## File-level change inventory

- `voice_input.py`:
  - Read `input_device`, `chunk_duration_s` from config.
  - Resolve `input_device` name → index; pass to `sd.InputStream`; log open.
  - In `toggle()`, on stop: `audio_stream.stop()`, transcribe, on next start `audio_stream.start()`.
  - In `transcribe_chunks`, branch on `chunk_duration_s == 0` for the single-call path.
  - Same wiring repeated in the `handsfree` branch where applicable (the VAD path also calls `transcribe_and_paste`, but the stream lifecycle is different — VAD has no explicit stop hotkey, so issue-2 fix is toggle-mode only).
- `setup.py`:
  - New "Microphone" step before model selection.
  - New "Chunk duration" step (advanced; offer `0` as default, allow integer override).
  - Persist new keys.
- `config.json`: new keys `input_device`, `chunk_duration_s`. Existing configs keep working — both keys are optional with safe defaults.
- `bench/` (new, gitignored except `bench/README.md` and scripts).
- `.gitignore`: add `bench/recordings/`.

## Open questions

None blocking. The chunking-upper-bound on Parakeet is empirical and the benchmark covers it.
