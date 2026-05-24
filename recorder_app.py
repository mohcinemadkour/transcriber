import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import threading
from datetime import datetime
import os
import time
from queue import Queue

# Page config
st.set_page_config(page_title="Simple Audio Recorder", layout="centered")
st.title("🎙️ Simple Audio Recorder")

SAMPLE_RATE = 16000
DEVICE_INDEX = 0

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.audio_data = None  # Store complete audio
        self.lock = threading.Lock()
        self.thread = None
        self.stream = None
    
    def record_thread_func(self):
        """Background recording thread - continuous recording"""
        print("[THREAD] Recording started")
        
        try:
            # Start continuous recording
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                device=DEVICE_INDEX,
                dtype=np.int16
            )
            self.stream.start()
            
            recorded_frames = []
            while self.is_recording:
                if not self.is_paused:
                    # Read available frames (non-blocking)
                    frames, overflowed = self.stream.read(8000)
                    if len(frames) > 0:
                        with self.lock:
                            recorded_frames.append(frames.copy())
                        max_val = np.max(np.abs(frames))
                        print(f"[THREAD] Read {len(frames)} frames, max: {max_val}")
                else:
                    time.sleep(0.1)
            
            self.stream.stop()
            self.stream.close()
            
            # Save complete audio
            if recorded_frames:
                with self.lock:
                    self.audio_data = np.concatenate(recorded_frames, axis=0)
                print(f"[THREAD] Finished - total audio: {len(self.audio_data)} samples")
            
        except Exception as e:
            print(f"[THREAD] Error: {e}")
    
    def start(self):
        """Start recording"""
        print("\n[APP] Starting recording...")
        self.is_recording = True
        self.is_paused = False
        self.audio_data = None
        self.thread = threading.Thread(target=self.record_thread_func, daemon=True)
        self.thread.start()
    
    def pause(self):
        """Toggle pause"""
        self.is_paused = not self.is_paused
        print(f"[APP] Recording {'paused' if self.is_paused else 'resumed'}")
    
    def stop(self):
        """Stop recording and return audio data"""
        print(f"\n[APP] Stopping recording...")
        self.is_recording = False
        
        if self.thread:
            print("[APP] Waiting for thread...")
            self.thread.join(timeout=3)
        
        print(f"[APP] Thread finished")
        
        if self.audio_data is not None and len(self.audio_data) > 0:
            print(f"[APP] Returning audio: {len(self.audio_data)} samples")
            return self.audio_data
        return None

# Initialize session state
if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder()
    st.session_state.status_msg = ""
    st.session_state.record_start_time = None
    print("[INIT] Created new recorder")

recorder = st.session_state.recorder

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start", use_container_width=True, key="start"):
        recorder.start()
        st.session_state.record_start_time = time.time()
        st.rerun()

with col2:
    if st.button("⏸️ Pause", use_container_width=True, key="pause"):
        if recorder.is_recording:
            recorder.pause()
            st.rerun()

with col3:
    if st.button("⏹️ Stop", use_container_width=True, key="stop"):
        if recorder.is_recording:
            audio_data = recorder.stop()
            st.session_state.record_start_time = None
            
            if audio_data is not None and len(audio_data) > 0:
                # Save file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                output_path = os.path.join(os.path.dirname(__file__), filename)
                
                print(f"[STOP] Saving to: {output_path}")
                wavfile.write(output_path, SAMPLE_RATE, audio_data)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    duration = len(audio_data) / SAMPLE_RATE
                    print(f"[STOP] Saved! Size: {file_size} bytes")
                    st.session_state.status_msg = f"✅ Saved: {filename} ({duration:.1f}s, {file_size/1024:.1f}KB)"
                else:
                    st.session_state.status_msg = "❌ Failed to save file"
            else:
                st.session_state.status_msg = "⚠️ No audio recorded"
            
            st.rerun()

# Status display with live updates
placeholder = st.empty()

if recorder.is_recording:
    with placeholder.container():
        # Calculate elapsed time
        if st.session_state.record_start_time:
            elapsed = time.time() - st.session_state.record_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        else:
            time_str = "0s"
        
        if recorder.is_paused:
            st.warning(f"⏸️ Paused ({time_str})")
        else:
            st.info(f"🔴 Recording... ({time_str})")
    
    # Auto-refresh every 0.5 seconds while recording
    time.sleep(0.5)
    st.rerun()
else:
    if st.session_state.status_msg:
        st.write(st.session_state.status_msg)
