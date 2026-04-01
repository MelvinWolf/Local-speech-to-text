"""
Voice Input — Main Script
Run setup.py first to configure.
"""
import sys
import json
import os
import subprocess
import ctypes
import ctypes.wintypes as wintypes
import threading
import time
import logging
import numpy as np
import sounddevice as sd
import pyautogui
import onnx_asr

# --- Single instance guard ---
mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "VoiceInputMutex")
if ctypes.windll.kernel32.GetLastError() == 183:
    sys.exit(0)

# --- Load config ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if not os.path.exists(CONFIG_FILE):
    print("No config.json found. Run setup.py first.")
    sys.exit(1)

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

MODE           = cfg.get("mode", "toggle")
HOTKEY_VK      = cfg.get("hotkey_vk", 0x6B)          # default: Numpad +
HOTKEY_NAME    = cfg.get("hotkey_name", "Numpad +")
VAD_SILENCE_MS = cfg.get("vad_silence_ms", 500)
MODEL_NAME     = cfg.get("model", "nemo-parakeet-tdt-0.6b-v3")
PROVIDERS      = cfg.get("providers", ["DmlExecutionProvider"])
SAMPLE_RATE    = 16000
CHUNK_DURATION = 20   # seconds — Parakeet/Whisper safe limit per chunk

# --- Logging ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_input.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger()

# --- Windows API ---
WH_KEYBOARD_LL = 13
WM_KEYDOWN     = 0x0100
WM_SYSKEYDOWN  = 0x0104

user32 = ctypes.windll.user32
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype  = wintypes.HHOOK
user32.CallNextHookEx.argtypes    = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype     = ctypes.c_long
user32.GetMessageW.argtypes       = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype        = wintypes.BOOL

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode",      wintypes.DWORD),
        ("scanCode",    wintypes.DWORD),
        ("flags",       wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

# --- Clipboard + paste ---
def set_clipboard(text):
    subprocess.run('clip', input=text.encode('utf-16-le'), check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)

def paste_clipboard():
    time.sleep(0.15)
    pyautogui.hotkey('ctrl', 'v')

# --- Model ---
log.info(f"Loading model: {MODEL_NAME}")
model = onnx_asr.load_model(MODEL_NAME, providers=PROVIDERS)
log.info("Model ready.")

# --- Audio ---
collecting   = False
audio_chunks = []

def audio_callback(indata, frames, time_info, status):
    if collecting:
        audio_chunks.append(indata.copy())

audio_stream = sd.InputStream(
    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
    callback=audio_callback
)
audio_stream.start()
log.info("Audio stream open.")

# --- Transcription ---
def transcribe_chunks(audio):
    chunk_size = SAMPLE_RATE * CHUNK_DURATION
    texts = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue
        result = model.recognize(chunk, sample_rate=SAMPLE_RATE)
        if result and result.strip():
            texts.append(result.strip())
    return " ".join(texts)

def transcribe_and_paste(audio_data=None):
    try:
        data = audio_data if audio_data is not None else audio_chunks
        if not data:
            return
        audio = np.concatenate(data).flatten()
        duration = len(audio) / SAMPLE_RATE
        log.info(f"Audio: {duration:.2f}s, max_amp={np.max(np.abs(audio)):.4f}")
        if len(audio) < SAMPLE_RATE * 0.3:
            log.info("Too short, ignored.")
            return
        n_chunks = int(np.ceil(duration / CHUNK_DURATION))
        log.info(f"Transcribing ({n_chunks} chunk(s))...")
        text = transcribe_chunks(audio)
        log.info(f"-> {text}")
        if text:
            if text[-1] in '.!?':
                text += ' '
            set_clipboard(text)
            paste_clipboard()
    except Exception as e:
        log.error(f"transcribe_and_paste error: {e}")

# ================================================================
# MODE: TOGGLE
# ================================================================
if MODE == "toggle":
    log.info(f"Mode: toggle  |  Hotkey: {HOTKEY_NAME} (VK 0x{HOTKEY_VK:02X})")

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

    HOOKPROC   = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    hook_handle = None

    def low_level_hook(nCode, wParam, lParam):
        if nCode >= 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if kb.vkCode == HOTKEY_VK:
                threading.Thread(target=toggle, daemon=True).start()
                return 1
        return user32.CallNextHookEx(hook_handle, nCode, wParam, lParam)

    hook_callback = HOOKPROC(low_level_hook)
    hook_handle   = user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_callback, None, 0)

    if not hook_handle:
        log.error("Failed to install keyboard hook")
        sys.exit(1)

    log.info("Hook installed. Listening...")

# ================================================================
# MODE: HANDS-FREE (VAD)
# ================================================================
elif MODE == "handsfree":
    log.info(f"Mode: hands-free  |  Silence threshold: {VAD_SILENCE_MS}ms")
    log.warning("VAD mode is experimental — may cut sentences short or miss speech. Toggle mode is more reliable.")

    ENERGY_THRESHOLD   = 0.02
    SILENCE_SAMPLES    = int(SAMPLE_RATE * (VAD_SILENCE_MS / 1000))
    MIN_SPEECH_SAMPLES = SAMPLE_RATE   # discard bursts shorter than 1s
    ONSET_CHUNKS       = 4             # consecutive speech frames required to start recording

    speech_active   = False
    onset_counter   = 0
    silence_counter = 0
    vad_chunks      = []

    def vad_audio_callback(indata, frames, time_info, status):
        global speech_active, onset_counter, silence_counter, vad_chunks

        rms       = float(np.sqrt(np.mean(indata ** 2)))
        is_speech = rms > ENERGY_THRESHOLD

        if is_speech:
            silence_counter = 0
            if not speech_active:
                onset_counter += 1
                vad_chunks.append(indata.copy())   # buffer onset frames
                if onset_counter >= ONSET_CHUNKS:
                    speech_active = True
                    onset_counter = 0
                    log.info("VAD: speech detected, recording...")
            else:
                vad_chunks.append(indata.copy())
        else:
            if onset_counter > 0:
                # never reached onset threshold — discard silently
                onset_counter = 0
                vad_chunks    = []
            if speech_active:
                silence_counter += frames
                vad_chunks.append(indata.copy())
                if silence_counter >= SILENCE_SAMPLES:
                    speech_active   = False
                    silence_counter = 0
                    total = sum(len(c) for c in vad_chunks)
                    if total >= MIN_SPEECH_SAMPLES:
                        snapshot = list(vad_chunks)   # immutable snapshot — no race condition
                        log.info("VAD: silence detected, transcribing...")
                        threading.Thread(
                            target=transcribe_and_paste, args=(snapshot,), daemon=True
                        ).start()
                    else:
                        log.info(f"VAD: too short ({total/SAMPLE_RATE:.2f}s), discarded.")
                    vad_chunks = []

    # Replace stream callback for VAD mode
    audio_stream.stop()
    audio_stream.close()
    audio_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=vad_audio_callback
    )
    audio_stream.start()
    log.info("VAD stream active. Listening...")

else:
    log.error(f"Unknown mode in config: {MODE}")
    sys.exit(1)

# --- Message pump ---
msg = wintypes.MSG()
while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
    ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
    ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
