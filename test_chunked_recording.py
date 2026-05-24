#!/usr/bin/env python3
"""
Test chunked recording like the Streamlit app does
"""
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from datetime import datetime

SAMPLE_RATE = 16000
DEVICE_INDEX = 0
CHUNK_SIZE = 8000  # 0.5 seconds
NUM_CHUNKS = 6     # ~3 seconds total

print(f"Recording {NUM_CHUNKS} chunks of {CHUNK_SIZE} samples ({NUM_CHUNKS * CHUNK_SIZE / SAMPLE_RATE:.1f}s)")
print("Speak into your microphone...")

chunks = []
for i in range(NUM_CHUNKS):
    chunk = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, 
                   device=DEVICE_INDEX, dtype=np.int16, blocksize=0)
    sd.wait()
    chunk_copy = chunk.copy()
    chunks.append(chunk_copy)
    
    max_level = np.max(np.abs(chunk_copy))
    print(f"Chunk {i+1}: max level = {max_level}")

print("\nCombining chunks...")
combined = np.concatenate(chunks, axis=0)
print(f"Combined shape: {combined.shape}")
print(f"Combined max level: {np.max(np.abs(combined))}")
print(f"Combined mean level: {np.mean(np.abs(combined))}")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"chunked_recording_{timestamp}.wav"
wavfile.write(filename, SAMPLE_RATE, combined)
print(f"\n✅ Saved: {filename}")
