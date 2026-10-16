# Voice Command Extension for Pi

This repository contains the **/voice** extension for the Pi coding agent. It provides a voice-to-text workflow:

1. Record audio from the microphone (PulseAudio, via `ffmpeg`).
2. Send the audio to a configurable Automatic Speech Recognition (ASR) backend (Whisper, Qwen, or local offline models).
3. Insert the transcription directly into the Pi editor.

## Features

- **VAD mode** – starts recording when speech is detected and automatically stops after a configurable silence period.
- **Fixed‑duration mode** – `/voice 10` records exactly 10 seconds.
- **Configurable parameters** – threshold, silenceMs, maxSeconds, leadInMs, source, model, cleanTranscript.
- **Presets** – quick tuning for headset, normal, or noisy environments.
- **Local‑offline models** – `local/sensevoice` and `local/whisper-base` for low‑latency transcription.
- **Debug dumps** – audio and raw ASR responses are saved under `~/.pi/agent/voice-debug` (private).

## Installation

### 1. Requirements

- **Pi** (pi-coding-agent) installed and working
- **Node 18+** (global `fetch`, `FormData`, `Blob`, `AbortSignal.timeout`)
- **`ffmpeg`** on the system PATH, with **PulseAudio** for microphone capture
- **An ASR backend** — either:
  - a remote OpenAI-compatible server (default `http://localhost:7999`, override with the `ASR_API_BASE` environment variable) serving at least one of the models listed in `ASR_MODELS` in the source (e.g. `openai/whisper-large-v3-turbo`, `Qwen/Qwen3-ASR-1.7B`), or
  - a local offline ASR project (see step 4)

### 2. Install the extension

```bash
git clone https://github.com/gucliti/voice-command-extension.git
cd voice-command-extension
mkdir -p ~/.pi/agent/extensions
cp voice-command.ts ~/.pi/agent/extensions/
```

Pi loads extensions from `~/.pi/agent/extensions/` automatically. Start Pi (or run `/reload` in a running session) — the `/voice` command and the `ctrl+alt+v` shortcut become available. No npm dependencies are needed.

### 3. First run

```text
/voice config noise   # measure ambient noise, get a suggested threshold
/voice config threshold <n>   # apply it if needed
/voice                # speak; transcript is pasted into the editor
```

Configuration is persisted in `~/.pi/agent/voice.json` (created on first write, hand-editable).

### 4. Optional: local offline ASR engines

The `local/sensevoice` and `local/whisper-base` models use a separate Python project. Set the `LOCAL_ASR_DIR` environment variable to its directory (default `~/.local/share/localASR`); it must contain `.venv/bin/python` and an `asr_file.py` that supports one-shot transcription (`asr_file.py <wav> --model <engine>`) and, optionally, a resident daemon mode (`--socket <path>`). Without it, only the remote Whisper/Qwen models are available.

### 5. Optional: sandboxed Pi

If you run Pi with filesystem sandboxing, allow writes to `~/.pi/agent/` (config file and debug dumps live there) and reads of the extension file itself.

## Usage

```text
/voice                 # VAD mode (default)
/voice 10              # Record exactly 10 seconds
/voice config          # Show current configuration
/voice config threshold 400   # Change a single key
/voice config preset headset   # Apply a preset
/voice load            # Warm‑up the ASR model in the background
```

All configuration is persisted in `~/.pi/agent/voice.json`.

---

## Contributing

Feel free to open issues or submit pull-requests. When adding new models, update the `ASR_MODELS` map in `voice-command.ts` and extend the documentation.
