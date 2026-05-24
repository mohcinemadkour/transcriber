#!/usr/bin/env python3
"""
Direct recording test - records audio and saves to WAV file
"""
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import os
from datetime import datetime

# Settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds
DEVICE_INDEX = 0  # Microsoft Sound Mapper

print(f"Recording test: {DURATION} seconds at {SAMPLE_RATE}Hz")
print(f"Device: {DEVICE_INDEX}")
print("Speak now...")

# Record audio
audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, 
                    channels=1, device=DEVICE_INDEX, dtype=np.int16, blocksize=0)
sd.wait()

# Check if audio was captured
max_level = np.max(np.abs(audio_data))
print(f"\nRecording complete!")
print(f"Samples captured: {len(audio_data)}")
print(f"Max level: {max_level}")

# Save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{timestamp}.wav"
output_path = os.path.join(os.path.dirname(__file__), filename)

if len(audio_data) > 0:
    wavfile.write(output_path, SAMPLE_RATE, audio_data)
    print(f"✅ Saved to: {output_path}")
    
    # Show file size
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / 1024:.1f} KB")
else:
    print("❌ No audio captured!")
