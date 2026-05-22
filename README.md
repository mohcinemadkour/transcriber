# System Audio Transcriber

Transcribe anything playing on your computer — meetings, videos, podcasts, music — using **faster-whisper** running fully locally (no API key, no cloud, no cost).

---

## Quick Start

```bash
# 1. Install dependencies
bash setup.sh          # Linux / macOS
pip install -r requirements.txt   # Windows (manual)

# 2. Set up loopback audio (see below)

# 3. Run
python transcriber.py
```

---

## How It Works

```
System Audio Output
       │
       ▼
[Virtual Loopback Device]    ← BlackHole / Stereo Mix / VB-Cable
       │
       ▼
 sounddevice (capture)
       │  5-second chunks  @ 16 kHz mono
       ▼
 faster-whisper (local)
       │  text segments
       ▼
  Rich TUI display  +  transcript_YYYYMMDD_HHMMSS.txt
```

---

## Loopback Setup (Required)

### macOS
1. Install **BlackHole 2ch**: https://existential.audio/blackhole/
2. Go to **System Settings → Sound → Output** → select **BlackHole 2ch**
3. *(Optional)* Create a **Multi-Output Device** in Audio MIDI Setup so you hear audio *and* capture it simultaneously.
4. In the app, select **BlackHole 2ch** as the input device.

### Windows
- **Option A — Stereo Mix** (free, built-in if supported):  
  Right-click speaker → Sounds → Recording → right-click empty area → *Show Disabled Devices* → enable **Stereo Mix**.
- **Option B — VB-Cable** (free): https://vb-audio.com/Cable/  
  Set Windows output to *CABLE Input*, select *CABLE Output* in the app.

### Linux
PulseAudio exposes a monitor source automatically:
```bash
pactl list sources short   # find *.monitor
```
Select that device in the app.

---

## Model Selection

| Model     | Size    | Speed  | Accuracy |
|-----------|---------|--------|----------|
| tiny      | 75 MB   | ████   | ★★☆☆☆   |
| base      | 145 MB  | ███    | ★★★☆☆   |
| small     | 465 MB  | ██     | ★★★★☆   |
| medium    | 1.5 GB  | █      | ★★★★★   |
| large-v3  | 3 GB    | ░      | ★★★★★+  |

`base` is the recommended default — fast enough for real-time on most laptops.

---

## Output

Each session saves a timestamped file:
```
transcript_20250522_143021.txt

[14:30:26] Hello and welcome to the podcast.
[14:30:31] Today we're talking about machine learning in production.
...
```

---

## Tips

- **Accuracy**: Use `small` or `medium` for lectures/meetings with technical vocabulary.
- **Latency**: Lower `CHUNK_SECONDS` in `transcriber.py` for faster but less accurate results.
- **Language**: Whisper auto-detects language; force it by setting `language="en"` in `engine.transcribe()`.
- **GPU**: If you have CUDA, change `device="cpu"` → `device="cuda"` and `compute_type="int8"` → `"float16"` for a major speedup.
