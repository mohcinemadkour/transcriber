"""
System Audio Transcriber
Captures loopback/system audio and transcribes it using faster-whisper (local, free).

Requirements:
  pip install faster-whisper sounddevice numpy rich

System audio capture setup (choose one):
  - macOS:   Install BlackHole (https://existential.audio/blackhole/)
             Set output to "BlackHole 2ch" (or multi-output device) in System Settings > Sound
  - Windows: Use "Stereo Mix" (enable in Sound > Recording devices)
             OR install VB-Cable (https://vb-audio.com/Cable/)
  - Linux:   Use PulseAudio monitor source (auto-detected as "monitor" device)
"""

import threading
import queue
import time
import sys
import numpy as np
import sounddevice as sd
from datetime import datetime
from pathlib import Path

# ── Rich TUI ──────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box
from rich.prompt import Prompt, Confirm

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # Whisper expects 16 kHz
CHUNK_SECONDS = 5        # seconds per transcription chunk
CHANNELS      = 1
DTYPE         = "float32"
WHISPER_MODEL = "base"   # tiny | base | small | medium | large-v3


# ── Audio capture ─────────────────────────────────────────────────────────────
class AudioCapture:
    def __init__(self, device_index: int | None = None):
        self.device_index = device_index
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer: list[np.ndarray] = []
        self._chunk_samples = SAMPLE_RATE * CHUNK_SECONDS
        self._stream: sd.InputStream | None = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # ignore overflow warnings in the callback
        self._buffer.append(indata.copy().flatten())
        total = sum(len(a) for a in self._buffer)
        if total >= self._chunk_samples:
            chunk = np.concatenate(self._buffer)[:self._chunk_samples]
            self._audio_queue.put(chunk)
            # keep leftovers
            leftover = np.concatenate(self._buffer)[self._chunk_samples:]
            self._buffer = [leftover] if len(leftover) else []

    def start(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self.device_index,
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ── Transcription engine ───────────────────────────────────────────────────────
class WhisperEngine:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self._model = None

    def load(self):
        from faster_whisper import WhisperModel
        console.print(f"[bold cyan]Loading Whisper model:[/bold cyan] [yellow]{self.model_name}[/yellow] (first run downloads it) …")
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        console.print("[bold green]✓ Model ready[/bold green]")

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""
        # faster-whisper wants float32 in [-1, 1]
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio /= 32768.0
        segments, _ = self._model.transcribe(audio, beam_size=5, language=None)
        return " ".join(seg.text.strip() for seg in segments).strip()


# ── Session log ───────────────────────────────────────────────────────────────
class TranscriptLog:
    def __init__(self):
        self.entries: list[dict] = []

    def add(self, text: str):
        if text:
            self.entries.append({"time": datetime.now().strftime("%H:%M:%S"), "text": text})

    def save(self, path: Path):
        with path.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(f"[{e['time']}] {e['text']}\n")
        return path


# ── Device listing ────────────────────────────────────────────────────────────
def list_audio_devices() -> list[dict]:
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append({"index": i, "name": d["name"], "hostapi": sd.query_hostapis(d["hostapi"])["name"]})
    return result


def pick_device() -> int | None:
    devices = list_audio_devices()
    table = Table(title="Available Input Devices", box=box.ROUNDED, border_style="cyan")
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("Device Name", style="white")
    table.add_column("Host API", style="dim")

    loopback_keywords = ["blackhole", "stereo mix", "monitor", "cable", "loopback", "vb-audio", "virtual"]
    for d in devices:
        is_loopback = any(k in d["name"].lower() for k in loopback_keywords)
        name_style = "bold green" if is_loopback else "white"
        suffix = " ← [bold green]recommended loopback[/bold green]" if is_loopback else ""
        table.add_row(str(d["index"]), Text(d["name"] + (" ★" if is_loopback else ""), style=name_style), d["hostapi"])

    console.print(table)
    console.print()
    console.print("[dim]Green ★ devices are likely loopback/system audio sources.[/dim]")
    console.print("[dim]Enter device number, or press Enter to use the default input device.[/dim]")
    console.print()

    choice = Prompt.ask("[bold cyan]Select device index[/bold cyan]", default="")
    if choice.strip() == "":
        return None
    try:
        idx = int(choice.strip())
        if any(d["index"] == idx for d in devices):
            return idx
        console.print("[red]Invalid index, using default.[/red]")
        return None
    except ValueError:
        return None


# ── Main TUI loop ─────────────────────────────────────────────────────────────
def run_transcriber(device_index: int | None, model_name: str):
    engine = WhisperEngine(model_name)
    engine.load()

    capture = AudioCapture(device_index)
    log = TranscriptLog()

    transcript_lines: list[str] = []
    status_text = "[bold yellow]● RECORDING[/bold yellow]"
    chunk_count = 0
    start_time = time.time()

    def make_display():
        elapsed = int(time.time() - start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        # Header
        header = Panel(
            Text("  SYSTEM AUDIO TRANSCRIBER  ", style="bold white on dark_cyan", justify="center"),
            subtitle=f"[dim]Model: {model_name}  |  Chunk: {CHUNK_SECONDS}s  |  SR: {SAMPLE_RATE} Hz[/dim]",
            border_style="cyan",
        )

        # Status bar
        status = Panel(
            Columns([
                Text(status_text),
                Text(f"[dim]Elapsed: {h:02d}:{m:02d}:{s:02d}[/dim]", justify="right"),
                Text(f"[dim]Chunks: {chunk_count}[/dim]", justify="right"),
            ]),
            box=box.SIMPLE,
        )

        # Transcript area — last 20 lines
        visible = transcript_lines[-20:]
        transcript_body = "\n".join(
            f"[dim]{log.entries[max(0, len(log.entries)-len(visible)+i)]['time']}[/dim]  {line}"
            if i < len(log.entries) else f"  {line}"
            for i, line in enumerate(visible)
        ) if visible else "[dim italic]Listening… audio chunks will appear here.[/dim italic]"

        transcript_panel = Panel(
            transcript_body,
            title="[bold]Transcript[/bold]",
            border_style="green",
            padding=(1, 2),
        )

        footer = Text(
            "  Ctrl+C to stop and save transcript  ",
            style="dim",
            justify="center",
        )

        from rich.console import Group
        return Group(header, status, transcript_panel, footer)

    capture.start()
    console.print(f"\n[bold green]Recording started.[/bold green] Speak or play audio. Press [bold]Ctrl+C[/bold] to stop.\n")

    def transcribe_worker():
        nonlocal chunk_count, status_text
        while True:
            chunk = capture.get_chunk(timeout=1.0)
            if chunk is None:
                continue
            status_text = "[bold yellow]◎ TRANSCRIBING…[/bold yellow]"
            text = engine.transcribe(chunk)
            chunk_count += 1
            status_text = "[bold green]● RECORDING[/bold green]"
            if text:
                log.add(text)
                transcript_lines.append(text)

    worker = threading.Thread(target=transcribe_worker, daemon=True)
    worker.start()

    try:
        with Live(make_display(), refresh_per_second=2, console=console) as live:
            while True:
                time.sleep(0.5)
                live.update(make_display())
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        console.print("\n[bold yellow]Stopping…[/bold yellow]")
        time.sleep(1)

        if log.entries:
            save_path = Path(f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            log.save(save_path)
            console.print(f"\n[bold green]✓ Transcript saved to:[/bold green] [cyan]{save_path.resolve()}[/cyan]")
            console.print(f"  [dim]{len(log.entries)} segments captured.[/dim]\n")
        else:
            console.print("\n[dim]No transcript content captured.[/dim]\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold white]System Audio Transcriber[/bold white]\n"
        "[dim]Powered by faster-whisper (local · free · private)[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

    # Model selection
    console.print("[bold]Whisper model sizes[/bold] [dim](accuracy vs speed trade-off):[/dim]")
    models = [
        ("tiny",     "~75 MB",  "Fastest, lower accuracy"),
        ("base",     "~145 MB", "Good balance  ← default"),
        ("small",    "~465 MB", "Better accuracy"),
        ("medium",   "~1.5 GB", "High accuracy"),
        ("large-v3", "~3 GB",   "Best accuracy, slowest"),
    ]
    t = Table(box=box.SIMPLE)
    t.add_column("Model", style="yellow")
    t.add_column("Size", style="dim")
    t.add_column("Notes")
    for m in models:
        t.add_row(*m)
    console.print(t)

    model_choice = Prompt.ask("[bold cyan]Choose model[/bold cyan]", default=WHISPER_MODEL,
                              choices=["tiny", "base", "small", "medium", "large-v3"])
    console.print()

    # Device selection
    device_index = pick_device()
    console.print()

    run_transcriber(device_index, model_choice)


if __name__ == "__main__":
    main()
