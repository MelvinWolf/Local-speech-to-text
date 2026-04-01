# Local Voice Input for Windows

Fast, private, local speech-to-text that types into any application. Press a hotkey to start recording, press again to stop — transcribed text is pasted at your cursor. Optionally runs fully hands-free via voice activity detection.

Built with [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) / [OpenAI Whisper](https://github.com/openai/whisper) via [onnx-asr](https://github.com/istupakov/onnx-asr), accelerated on AMD GPUs through DirectML. No cloud, no subscription, no data leaving your machine.

---

## Who this is for

**Hardware**
- Windows 10 or 11 (64-bit) — the keyboard hook and clipboard APIs used here are Windows-specific
- AMD GPU (RX 500 series or newer) for DirectML acceleration, OR any modern CPU for CPU-only inference
- NVIDIA users: this works but [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CUDA is a better stack for you
- 8 GB+ RAM free at runtime (the default model loads ~2.5 GB)

**Use case**
- Dictating into browsers, chat apps, editors, Office — anything that accepts Ctrl+V
- Primarily English; multilingual models support DE/FR/ES/IT with good accuracy
- Comfortable with ~300–500 ms latency after stopping (not real-time word-by-word streaming): Faster tha
- Privacy-conscious users who don't want audio sent to cloud APIs

**Not suitable for**
- UAC dialogs, games with anti-cheat, or any window that blocks synthetic keyboard input
- Real-time captioning or live transcription overlays
- macOS or Linux (would require significant rewrites)
- Locked-down corporate machines where kernel-level keyboard hooks are blocked by EDR/Group Policy

---

## Installation

### 1. Install Python

Download Python 3.10–3.13 from [python.org](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.

Verify:
```powershell
python --version
```

### 2. Clone or download this repo

```powershell
git clone https://github.com/MelvinWolf/Local-speech-to-text.git
cd local-speech-to-text
```

Or download the ZIP and extract it.

### 3. Install dependencies

> **Important:** `onnxruntime` and `onnxruntime-directml` conflict. If you have `onnxruntime` already installed, uninstall it first.

```powershell
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-directml
pip install onnx-asr huggingface_hub sounddevice numpy pyautogui
```

> **Note:** `pip install -r requirements.txt` will not handle the `onnxruntime` conflict automatically. Always run the `pip uninstall` line above first if `onnxruntime` or `onnxruntime-gpu` is already installed on your system.

Verify DirectML is available:
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
You should see `DmlExecutionProvider` in the list. If you only see `CPUExecutionProvider`, your GPU driver may not support DirectML — select CPU in the setup wizard.

### 4. Run the setup wizard

```powershell
python setup.py
```

The wizard will ask for your preferences and save them to `config.json`. Re-run it anytime to change settings.

### 5. Start voice input

```powershell
python voice_input.py
```

To run silently in the background on Windows startup, see [Auto-start](#auto-start).

---

## Setup wizard options

### Input mode

**Toggle** *(recommended)*
Press your hotkey once to start recording, press again to stop and transcribe. Good if you pause mid-thought — recording continues until you explicitly stop.

**Hands-free**
Voice activity detection (VAD) monitors your microphone continuously. When you start speaking, recording begins automatically. When silence is detected for the configured duration, transcription triggers. No button needed, but you cannot pause mid-sentence without triggering a commit.

> **Known limitations of hands-free mode:** The energy-based VAD can cut recordings short during natural mid-sentence pauses, struggle with background noise, and occasionally miss the beginning of speech. These are inherent limitations of simple energy-threshold detection. Toggle mode is more reliable for most use cases. Contributions implementing a proper ML-based VAD are welcome.

### Hotkey *(toggle mode only)*

The wizard listens for a physical key press and captures the Windows virtual key code. Any key works — function keys, numpad keys, media keys. Numpad + (`VK 0x6B`) is a good default if you have a numpad, as it doesn't interfere with normal typing.

### Silence delay *(hands-free mode only)*

How long silence must persist (in milliseconds) before transcription is triggered. Default is 500 ms. Increase this if you pause frequently mid-sentence; decrease it for faster response at the cost of cutting off trailing words.

### Model

| Model | Size | Languages | Best for |
|---|---|---|---|
| **Parakeet v3** *(default)* | 2.5 GB | EN, DE, FR, ES + more | Best balance — recommended for European multilingual use |
| **Parakeet v2** | 1.2 GB | English only | Fastest inference, English-only workflows |
| **Canary 1B** | 4 GB | EN, DE, FR, ES, IT + more | Highest multilingual accuracy for European languages |
| **Whisper large-v3-turbo** | 3 GB | 99 languages | Best overall multilingual, widest language coverage |
| **Whisper tiny** | 150 MB | 99 languages | CPU-only machines or very low-end hardware; accuracy is noticeably lower |
| **Moonshine tiny** | 100 MB | English only | Fastest possible, lightest footprint, English only |

Models are downloaded automatically from Hugging Face on first use. Subsequent runs load from local cache.

### Execution provider

| Option | When to use |
|---|---|
| **DirectML** *(default)* | AMD or Intel GPU on Windows |
| **CUDA** | NVIDIA GPU | ENTIRELY UNTESTED AS OF NOW |
| **CPU** | No GPU, or if GPU acceleration causes issues |

---

## Files

| File | Purpose |
|---|---|
| `setup.py` | Interactive wizard — run once to configure, re-run to change settings |
| `voice_input.py` | Main script — reads `config.json` and runs voice input |
| `config.json` | Generated by `setup.py`, human-readable, safe to edit manually |
| `voice_input.log` | Runtime log — check here if something isn't working |
| `voice_input_startup.vbs` | Optional launcher for silent background auto-start |

---

## Auto-start

To have voice input start silently on login with no console window:

1. Edit `voice_input_startup.vbs` and update the path to match where you placed `voice_input.py`
2. Press `Win + R`, type `shell:startup`, press Enter
3. Copy `voice_input_startup.vbs` into that folder

Voice input will now launch automatically on every login. To stop it, open Task Manager and end `pythonw.exe`.

---

## Troubleshooting

**Transcription is empty / very low amplitude in log**
Multiple instances may be running and interfering with the audio device. Open Task Manager, end all `python.exe` and `pythonw.exe` processes, then relaunch.

**Wrong microphone being used**
`sounddevice` picks the Windows default input device. If that's not your mic, open Windows Sound Settings → Input → set your preferred microphone as default.

**Antivirus warning or hook failing**
Some EDR/AV products flag low-level keyboard hooks (`WH_KEYBOARD_LL`). Add the folder to your AV exclusion list, or run as administrator to confirm the hook installs correctly.

**`DmlExecutionProvider` not appearing**
Update your GPU drivers. DirectML requires relatively recent drivers on AMD (Adrenalin 21.x+) and Intel. If it still doesn't appear, select CPU in the setup wizard.

**Hands-free mode triggering on background noise**
The energy threshold (`ENERGY_THRESHOLD = 0.02` in `voice_input.py`) may need raising. Open the file and increase the value, or use toggle mode instead.

---

## License

MIT
