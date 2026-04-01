"""
Voice Input Setup Wizard
Run this once to configure voice_input.py
"""
import json
import ctypes
import ctypes.wintypes as wintypes
import os
import sys
import time

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

MODELS = [
    {
        "key": "nemo-parakeet-tdt-0.6b-v3",
        "label": "Parakeet v3 — multilingual EN/DE/FR/ES  [Recommended, 2.5GB]"
    },
    {
        "key": "nemo-parakeet-tdt-0.6b-v2",
        "label": "Parakeet v2 — English only, fastest  [1.2GB]"
    },
    {
        "key": "nemo-canary-1b-flash",
        "label": "Canary 1B — best EU multilingual accuracy  [4GB]"
    },
    {
        "key": "openai-whisper-large-v3-turbo",
        "label": "Whisper large-v3-turbo — best overall multilingual  [3GB]"
    },
    {
        "key": "openai-whisper-tiny",
        "label": "Whisper tiny — CPU-friendly, lowest quality  [150MB]"
    },
    {
        "key": "moonshine-tiny",
        "label": "Moonshine tiny — fastest English-only, lightest  [100MB]"
    },
]

PROVIDERS = [
    ("DmlExecutionProvider",  "AMD / Intel GPU via DirectML  [Recommended for non-NVIDIA]"),
    ("CUDAExecutionProvider", "NVIDIA GPU via CUDA"),
    ("CPUExecutionProvider",  "CPU only  [Slowest, universal]"),
]

# --- Windows hook for key detection ---
WH_KEYBOARD_LL = 13
WM_KEYDOWN     = 0x0100

user32 = ctypes.windll.user32
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype  = wintypes.HHOOK
user32.CallNextHookEx.argtypes    = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype     = ctypes.c_long
user32.GetMessageW.argtypes       = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetMessageW.restype        = wintypes.BOOL
user32.PostQuitMessage.argtypes   = [ctypes.c_int]
user32.PostQuitMessage.restype    = None

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode",      wintypes.DWORD),
        ("scanCode",    wintypes.DWORD),
        ("flags",       wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

def capture_keypress():
    """Block until user presses a key, return (vk_code, key_name)."""
    captured = {}
    hook_handle = None
    HOOKPROC = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

    def hook(nCode, wParam, lParam):
        if nCode >= 0 and wParam == WM_KEYDOWN:
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            captured["vk"] = kb.vkCode
            user32.PostQuitMessage(0)
            return 1
        return user32.CallNextHookEx(hook_handle, nCode, wParam, lParam)

    cb = HOOKPROC(hook)
    hook_handle = user32.SetWindowsHookExW(WH_KEYBOARD_LL, cb, None, 0)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        pass
    user32.UnhookWindowsHookEx(hook_handle)
    return captured.get("vk", 0)

def menu(prompt, options):
    """Display numbered menu, return 0-based index."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        label = opt if isinstance(opt, str) else opt[1]
        print(f"  {i}. {label}")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
            if 1 <= choice <= len(options):
                return choice - 1
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid choice, try again.")

def vk_to_name(vk):
    """Best-effort human-readable name for a VK code."""
    names = {
        0x6B: "Numpad +", 0x6D: "Numpad -", 0x6A: "Numpad *",
        0x6F: "Numpad /", 0x0D: "Enter", 0x20: "Space",
        0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4",
        0x74: "F5", 0x75: "F6", 0x76: "F7", 0x77: "F8",
        0x78: "F9", 0x79: "F10", 0x7A: "F11", 0x7B: "F12",
        0x13: "Pause", 0x91: "Scroll Lock",
    }
    if vk in names:
        return names[vk]
    if 0x41 <= vk <= 0x5A:
        return chr(vk)
    return f"VK 0x{vk:02X}"

def main():
    print("=" * 55)
    print("  Voice Input — Setup Wizard")
    print("=" * 55)

    config = {}

    # Load existing config as defaults if present
    existing = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                existing = json.load(f)
            print(f"\nExisting config found. Press Enter to keep current value or enter new one.")
        except Exception:
            pass

    # --- Mode ---
    mode_idx = menu(
        "Input mode:",
        ["Toggle  — press hotkey to start, press again to stop  [Recommended]",
         "Hands-free  — speaks detected automatically via VAD, no button needed"]
    )
    config["mode"] = ["toggle", "handsfree"][mode_idx]

    # --- Hotkey (toggle mode only) ---
    if config["mode"] == "toggle":
        print("\nHotkey detection:")
        print("  Press the key you want to use as your toggle hotkey...")
        vk = capture_keypress()
        name = vk_to_name(vk)
        print(f"  Captured: {name}  (VK 0x{vk:02X})")
        config["hotkey_vk"]   = vk
        config["hotkey_name"] = name

    # --- VAD silence delay (handsfree mode only) ---
    if config["mode"] == "handsfree":
        default_ms = existing.get("vad_silence_ms", 500)
        try:
            val = input(f"\nSilence duration before auto-commit in ms  [default {default_ms}]: ").strip()
            config["vad_silence_ms"] = int(val) if val else default_ms
        except ValueError:
            config["vad_silence_ms"] = default_ms

    # --- Model ---
    model_idx = menu("Model:", [m["label"] for m in MODELS])
    config["model"] = MODELS[model_idx]["key"]

    # --- Provider ---
    provider_idx = menu("Execution provider (GPU/CPU backend):", [p[1] for p in PROVIDERS])
    config["providers"] = [PROVIDERS[provider_idx][0]]

    # --- Save ---
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to: {CONFIG_FILE}")
    print("\nSummary:")
    print(f"  Mode     : {config['mode']}")
    if config["mode"] == "toggle":
        print(f"  Hotkey   : {config['hotkey_name']} (VK 0x{config['hotkey_vk']:02X})")
    else:
        print(f"  VAD delay: {config['vad_silence_ms']}ms")
    print(f"  Model    : {config['model']}")
    print(f"  Provider : {config['providers'][0]}")
    print("\nRun voice_input.py to start. Re-run setup.py anytime to change settings.")

if __name__ == "__main__":
    main()
