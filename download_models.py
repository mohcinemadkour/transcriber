#!/usr/bin/env python3
"""
Download all Whisper models locally for offline use
This ensures models are cached and don't need to download on first transcription
"""

from faster_whisper import WhisperModel
import os

models = ["tiny", "base", "small", "medium", "large-v3"]

print("🚀 Downloading Whisper models locally...")
print("=" * 60)

for model_name in models:
    try:
        print(f"\n📥 Downloading {model_name.upper()} model...")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"✅ {model_name.upper()} model ready!")
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {str(e)}")

print("\n" + "=" * 60)
print("✅ All models downloaded successfully!")
print("💾 Models are cached locally - no need to download again")
print("\nCache location:")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"   {cache_dir}")
