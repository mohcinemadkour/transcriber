#!/usr/bin/env python3
"""
Test recording with different devices to debug audio capture
"""
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import os
from datetime import datetime

SAMPLE_RATE = 16000
DURATION = 3

print("Available audio devices:")
print(sd.query_devices())

print("\n" + "="*60)
print("Testing Device 0 (Default/Microsoft Sound Mapper)")
print("="*60)

try:
    print(f"Recording {DURATION} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, 
                   channels=1, device=0, dtype=np.int16, blocksize=0)
    sd.wait()
    
    # Check audio levels
    max_level = np.max(np.abs(audio))
    min_level = np.min(np.abs(audio))
    mean_level = np.mean(np.abs(audio))
    
    print(f"\nRecording complete!")
    print(f"Samples: {len(audio)}")
    print(f"Max level: {max_level}")
    print(f"Min level: {min_level}")
    print(f"Mean level: {mean_level}")
    
    # Save
    if max_level > 100:  # Good signal
        print("✅ Audio detected! Saving...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_recording_{timestamp}.wav"
        wavfile.write(filename, SAMPLE_RATE, audio)
        print(f"Saved: {filename}")
    else:
        print("⚠️ Very low audio levels (may be silent)")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_recording_silent_{timestamp}.wav"
        wavfile.write(filename, SAMPLE_RATE, audio)
        print(f"Saved anyway: {filename}")
        
except Exception as e:
    print(f"❌ Error: {e}")
