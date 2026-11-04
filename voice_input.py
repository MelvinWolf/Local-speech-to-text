"""
Voice Input — Main Script
Run setup.py first to configure.
"""
import sys
import json
import os
import re
import subprocess
import ctypes
import ctypes.wintypes as wintypes
import threading
import time
import logging
import winsound

# --- Ensure attached to interactive 'Default' desktop (before importing pyautogui) ---
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32   = ctypes.windll.user32
_DEFAULT_HDESK = user32.OpenDesktopW("Default", 0, False, 0x10000000)

def _attach_default_desktop():
    """Attach the calling thread to WinSta0\\Default if currently on a non-default desktop."""
    if _DEFAULT_HDESK:
        user32.SetThreadDesktop(_DEFAULT_HDESK)

def _get_current_desktop_name():
    buf = ctypes.create_unicode_buffer(256)
    needed = wintypes.DWORD()
    hdesk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
    user32.GetUserObjectInformationW(hdesk, 2, buf, ctypes.sizeof(buf), ctypes.byref(needed))
    return buf.value

_attach_default_desktop()

import numpy as np
import sounddevice as sd
import pyautogui
import onnxruntime as ort
import onnx_asr

# --- Single instance guard ---
mutex = kernel32.CreateMutexW(None, False, "VoiceInputMutex")
if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
    sys.exit(0)

# --- Load config ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if not os.path.exists(CONFIG_FILE):
    print("No config.json found. Run setup.py first.")
    sys.exit(1)

with open(CONFIG_FILE) as f:
    cfg = json.load(f)

LEGACY_MODEL_MAP = {
    "nemo-canary-1b-flash": "nemo-canary-1b-v2",
    "openai-whisper-large-v3-turbo": "onnx-community/whisper-large-v3-turbo",
    "openai-whisper-tiny": "onnx-community/whisper-tiny",
}

MODE               = cfg.get("mode", "toggle")
HOTKEY_VK          = cfg.get("hotkey_vk", 0x6B)          # default: Numpad +
HOTKEY_NAME        = cfg.get("hotkey_name", "Numpad +")
VAD_SILENCE_MS     = cfg.get("vad_silence_ms", 500)
RAW_MODEL_NAME     = cfg.get("model", "nemo-parakeet-tdt-0.6b-v3")
MODEL_NAME         = LEGACY_MODEL_MAP.get(RAW_MODEL_NAME, RAW_MODEL_NAME)
QUANTIZATION       = cfg.get("quantization", None)
PROVIDERS          = cfg.get("providers", ["DmlExecutionProvider"])
INPUT_DEVICE       = cfg.get("input_device")             # human-readable device name, optional
CHUNK_DURATION_S   = cfg.get("chunk_duration_s", 0)      # 0 = no chunking, feed whole audio in one call
AUDIO_FEEDBACK     = cfg.get("audio_feedback", True)     # subtle beep on record start/stop
PRESERVE_CLIPBOARD = cfg.get("preserve_clipboard", True) # restore previous clipboard after paste
SAMPLE_RATE        = 16000

# --- Logging ---
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_input.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger()

# --- Windows API ---
WH_KEYBOARD_LL = 13
WM_KEYDOWN     = 0x0100
WM_KEYUP       = 0x0101
WM_SYSKEYDOWN  = 0x0104
WM_SYSKEYUP    = 0x0105
CF_UNICODETEXT = 13

user32 = ctypes.windll.user32
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype  = wintypes.HHOOK
user32.CallNextHookEx.argtypes    = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype     = ctypes.c_long
user32.GetMessageW.argtypes       = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype        = wintypes.BOOL

user32.OpenClipboard.argtypes     = [wintypes.HWND]
user32.OpenClipboard.restype      = wintypes.BOOL
user32.CloseClipboard.argtypes    = []
user32.CloseClipboard.restype     = wintypes.BOOL
user32.GetClipboardData.argtypes  = [wintypes.UINT]
user32.GetClipboardData.restype   = wintypes.HANDLE
kernel32.GlobalLock.argtypes      = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype       = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes    = [wintypes.HGLOBAL]
kernel32.GlobalUnlock.restype     = wintypes.BOOL

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode",      wintypes.DWORD),
        ("scanCode",    wintypes.DWORD),
        ("flags",       wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

# --- Clipboard + paste ---
def remove_fillers(text):
    """Remove filler words 'um' and 'uh' as whole words, handling all punctuation contexts."""
    text = re.sub(r',\s*\b(um|uh)\b\s*,', ',',   text, flags=re.IGNORECASE)  # , um, → ,
    text = re.sub(r'^\s*\b(um|uh)\b\s*,?\s*', '', text, flags=re.IGNORECASE)  # start
    text = re.sub(r',?\s*\b(um|uh)\b\s*$', '',   text, flags=re.IGNORECASE)  # end
    text = re.sub(r'\b(um|uh)\b', '',             text, flags=re.IGNORECASE)  # middle
    text = re.sub(r' {2,}', ' ',                  text)                       # double spaces
    text = re.sub(r' ([,\.!?;:])', r'\1',          text)                       # space before punct
    return text.strip()

def get_clipboard_text():
    """Return current clipboard text if available, else None."""
    if not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    except Exception:
        return None
    finally:
        user32.CloseClipboard()

def set_clipboard(text):
    subprocess.run('clip', input=text.encode('utf-16-le'), check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)

def paste_text(text):
    _attach_default_desktop()
    prev_clip = get_clipboard_text() if PRESERVE_CLIPBOARD else None
    set_clipboard(text)
    time.sleep(0.08)
    pyautogui.hotkey('ctrl', 'v')
    if PRESERVE_CLIPBOARD and prev_clip is not None:
        time.sleep(0.15)
        try:
            set_clipboard(prev_clip)
        except Exception:
            pass

def play_cue(start=True):
    """Play a brief, non-blocking tone on record start/stop if audio_feedback is enabled."""
    if not AUDIO_FEEDBACK:
        return
    def _beep():
        try:
            winsound.Beep(880 if start else 520, 45)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

# --- Model ---
def _load_asr_model():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    use_dml = any(
        (p == "DmlExecutionProvider" if isinstance(p, str) else p[0] == "DmlExecutionProvider")
        for p in PROVIDERS
    )
    preproc_cfg = {"use_numpy_preprocessors": True} if use_dml else None
    m = onnx_asr.load_model(
        MODEL_NAME,
        quantization=QUANTIZATION,
        sess_options=so,
        providers=PROVIDERS,
        preprocessor_config=preproc_cfg,
    )
    # Warmup pass: compiles DirectML shaders at startup and verifies ONNX Runtime health
    m.recognize(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), sample_rate=SAMPLE_RATE)
    return m

log.info(f"Loading model: {MODEL_NAME} (quantization={QUANTIZATION}, providers={PROVIDERS})")
model = _load_asr_model()
log.info("Model ready and warmed up.")

# --- Device resolution ---
def _resolve_input_device(name, devices):
    """
    Return the index of the best input-capable device matching `name`.
    Handles 31-character MME truncation via prefix matching and prefers
    lower-latency host APIs (WASAPI / DirectSound) over MME when multiple match.
    """
    if not name:
        return None
    exact = [
        (i, dev) for i, dev in enumerate(devices)
        if dev.get("name") == name and dev.get("max_input_channels", 0) > 0
    ]
    if exact:
        exact.sort(key=lambda item: item[1].get("hostapi", 0), reverse=True)
        return exact[0][0]
    # Fallback: MME truncates device names to 31 chars
    prefix = [
        (i, dev) for i, dev in enumerate(devices)
        if dev.get("max_input_channels", 0) > 0 and (
            dev.get("name", "").startswith(name) or name.startswith(dev.get("name", "---"))
        )
    ]
    if prefix:
        prefix.sort(key=lambda item: item[1].get("hostapi", 0), reverse=True)
        return prefix[0][0]
    return None

# --- Audio ---
collecting   = False
audio_chunks = []
_toggle_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    if collecting:
        audio_chunks.append(indata.copy())

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
try:
    _opened = sd.query_devices(_device_idx if _device_idx is not None else sd.default.device[0])
    log.info(f"Audio stream open on device: {_opened['name']!r} (hostapi {_opened['hostapi']})")
except Exception as e:
    log.info(f"Audio stream open (device introspection failed: {e})")

# --- Transcription ---
def _is_gpu_lost(e):
    msg = str(e).lower()
    return "device" in msg and ("suspended" in msg or "removed" in msg)

def _recognize_with_recovery(audio):
    """Run model.recognize; on a transient GPU-suspended/removed error, reload the model and retry once."""
    global model
    try:
        return model.recognize(audio, sample_rate=SAMPLE_RATE)
    except Exception as e:
        if not _is_gpu_lost(e):
            raise
        log.warning(f"GPU device lost during recognize; reloading model and retrying. Original error: {e}")
        model = _load_asr_model()
        return model.recognize(audio, sample_rate=SAMPLE_RATE)

def transcribe_chunks(audio):
    if CHUNK_DURATION_S <= 0:
        result = _recognize_with_recovery(audio)
        return result.strip() if result else ""
    chunk_size = SAMPLE_RATE * CHUNK_DURATION_S
    texts = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < SAMPLE_RATE * 0.3:
            continue
        result = _recognize_with_recovery(chunk)
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
        t0 = time.perf_counter()
        if CHUNK_DURATION_S > 0:
            n_chunks = int(np.ceil(duration / CHUNK_DURATION_S))
            log.info(f"Transcribing ({n_chunks} chunk(s))...")
        else:
            log.info("Transcribing (single call)...")
        text = remove_fillers(transcribe_chunks(audio))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.info(f"-> {text}  ({elapsed_ms:.0f}ms)")
        if text:
            if text[-1] in '.!?':
                text += ' '
            paste_text(text)
    except Exception as e:
        log.error(f"transcribe_and_paste error: {e}")

# ================================================================
# MODE: TOGGLE / HOLD (PUSH-TO-TALK)
# ================================================================
if MODE in ("toggle", "hold"):
    log.info(f"Mode: {MODE}  |  Hotkey: {HOTKEY_NAME} (VK 0x{HOTKEY_VK:02X})")

    def start_recording():
        global collecting, audio_chunks
        with _toggle_lock:
            if collecting:
                return
            audio_chunks = []
            if not audio_stream.active:
                audio_stream.start()
            collecting = True
            log.info("Recording...")
            play_cue(start=True)

    def stop_recording():
        global collecting
        with _toggle_lock:
            if not collecting:
                return
            # Stop FIRST so PortAudio drains pending buffers into the callback
            # while the gate is still open. Closing the gate first would cause
            # those flushed callbacks to drop their audio.
            audio_stream.stop()
            collecting = False
            snapshot = list(audio_chunks)
            # Resume the stream now so the next start hotkey is instant.
            audio_stream.start()
            play_cue(start=False)
            threading.Thread(
                target=transcribe_and_paste, args=(snapshot,), daemon=True
            ).start()

    def toggle():
        try:
            if not collecting:
                start_recording()
            else:
                stop_recording()
        except Exception as e:
            log.error(f"toggle error: {e}")

    HOOKPROC     = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    hook_handle  = None
    _hotkey_down = False

    def low_level_hook(nCode, wParam, lParam):
        global _hotkey_down
        if nCode >= 0:
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if kb.vkCode == HOTKEY_VK:
                if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    if not _hotkey_down:
                        _hotkey_down = True
                        if MODE == "toggle":
                            threading.Thread(target=toggle, daemon=True).start()
                        else:
                            threading.Thread(target=start_recording, daemon=True).start()
                    return 1
                elif wParam in (WM_KEYUP, WM_SYSKEYUP):
                    _hotkey_down = False
                    if MODE == "hold":
                        threading.Thread(target=stop_recording, daemon=True).start()
                    return 1
        return user32.CallNextHookEx(hook_handle, nCode, wParam, lParam)

    hook_callback = HOOKPROC(low_level_hook)
    hook_handle   = user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_callback, None, 0)

    if not hook_handle:
        log.error("Failed to install keyboard hook")
        sys.exit(1)

    log.info(f"Hook installed on desktop {_get_current_desktop_name()!r}. Listening...")

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
        callback=vad_audio_callback,
        device=_device_idx,
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
