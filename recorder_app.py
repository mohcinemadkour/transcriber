import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import threading
from datetime import datetime
import os
import time

# Page config
st.set_page_config(page_title="Audio Recorder", layout="centered")
st.title("🎙️ Audio Recorder")

SAMPLE_RATE = 16000

# Auto-detect best recording devices
def get_best_devices():
    """Auto-detect best microphone and system audio devices"""
    devices = sd.query_devices()
    
    mic_device = None
    media_device = None
    
    # Priority order for microphone
    mic_keywords = ['microphone', 'mic input', 'array', 'mapper']
    # Priority order for system audio
    media_keywords = ['stereo mix', 'what u hear', 'wave out', 'loopback']
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            name_lower = device['name'].lower()
            
            # Find best microphone (first match)
            if mic_device is None:
                for keyword in mic_keywords:
                    if keyword in name_lower:
                        mic_device = i
                        break
            
            # Find best system audio (first match)
            if media_device is None:
                for keyword in media_keywords:
                    if keyword in name_lower:
                        media_device = i
                        break
    
    # Fallback: use device 0 if nothing found
    if mic_device is None:
        mic_device = 0
    
    return mic_device, media_device

mic_device, media_device = get_best_devices()
print(f"[INIT] Detected Microphone: Device {mic_device}, System Audio: Device {media_device}")

class AudioRecorder:
    def __init__(self, mode="microphone"):
        """mode: 'microphone', 'media', or 'both'"""
        self.mode = mode
        self.is_recording = False
        self.is_paused = False
        self.audio_data = None
        self.lock = threading.Lock()
        self.thread = None
        self.streams = {}
    
    def record_thread_func(self):
        """Background recording thread"""
        print(f"[THREAD] Recording started - mode: {self.mode}")
        
        try:
            recorded_frames = {}
            
            # Setup streams based on mode
            if self.mode in ["microphone", "both"]:
                recorded_frames["mic"] = []
                self.streams["mic"] = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, 
                    device=mic_device, dtype=np.int16
                )
                self.streams["mic"].start()
            
            if self.mode in ["media", "both"] and media_device is not None:
                recorded_frames["media"] = []
                self.streams["media"] = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1,
                    device=media_device, dtype=np.int16
                )
                self.streams["media"].start()
            
            # Recording loop
            while self.is_recording:
                if not self.is_paused:
                    for stream_name, stream in self.streams.items():
                        frames, _ = stream.read(8000)
                        if len(frames) > 0:
                            with self.lock:
                                recorded_frames[stream_name].append(frames.copy())
                else:
                    time.sleep(0.1)
            
            # Close streams
            for stream in self.streams.values():
                stream.stop()
                stream.close()
            
            # Combine audio
            if self.mode == "both" and len(recorded_frames) == 2:
                # Mix microphone and media
                mic_audio = np.concatenate(recorded_frames["mic"], axis=0)
                media_audio = np.concatenate(recorded_frames["media"], axis=0)
                
                # Match lengths
                min_len = min(len(mic_audio), len(media_audio))
                mic_audio = mic_audio[:min_len]
                media_audio = media_audio[:min_len]
                
                # Mix (average)
                self.audio_data = (mic_audio.astype(np.float32) + media_audio.astype(np.float32)) / 2
                self.audio_data = self.audio_data.astype(np.int16)
                print(f"[THREAD] Mixed audio: {len(self.audio_data)} samples")
            else:
                # Single source
                for stream_name in recorded_frames:
                    if recorded_frames[stream_name]:
                        with self.lock:
                            self.audio_data = np.concatenate(recorded_frames[stream_name], axis=0)
                        print(f"[THREAD] {stream_name} audio: {len(self.audio_data)} samples")
                        break
            
        except Exception as e:
            print(f"[THREAD] Error: {e}")
    
    def start(self):
        """Start recording"""
        print(f"\n[APP] Starting {self.mode} recording...")
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
        """Stop recording"""
        print(f"\n[APP] Stopping recording...")
        self.is_recording = False
        
        if self.thread:
            self.thread.join(timeout=3)
        
        if self.audio_data is not None and len(self.audio_data) > 0:
            print(f"[APP] Returning audio: {len(self.audio_data)} samples")
            return self.audio_data
        return None

# Initialize session state
if "recorder" not in st.session_state:
    st.session_state.recorder = None
    st.session_state.status_msg = ""
    st.session_state.record_start_time = None
    st.session_state.record_mode = "microphone"

# Recording mode selection
st.subheader("🎵 Recording Mode")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎤 Microphone", use_container_width=True, key="mode_mic"):
        if st.session_state.recorder is None or not st.session_state.recorder.is_recording:
            st.session_state.record_mode = "microphone"
            st.rerun()

with col2:
    media_disabled = media_device is None
    if st.button("🔊 Media", use_container_width=True, key="mode_media", disabled=media_disabled):
        if st.session_state.recorder is None or not st.session_state.recorder.is_recording:
            st.session_state.record_mode = "media"
            st.rerun()
    
    if media_disabled:
        st.caption("⚠️ Not available")

with col3:
    both_disabled = media_device is None
    if st.button("🎙️ Both", use_container_width=True, key="mode_both", disabled=both_disabled):
        if st.session_state.recorder is None or not st.session_state.recorder.is_recording:
            st.session_state.record_mode = "both"
            st.rerun()
    
    if both_disabled:
        st.caption("⚠️ Needs media")

# Show selected mode
mode_labels = {
    "microphone": "🎤 Microphone Only",
    "media": "🔊 System Audio Only",
    "both": "🎙️ Microphone + System Audio"
}
st.info(f"**Recording Mode:** {mode_labels[st.session_state.record_mode]}")

# Recording controls
st.subheader("🎯 Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start", use_container_width=True, key="start"):
        st.session_state.recorder = AudioRecorder(mode=st.session_state.record_mode)
        st.session_state.recorder.start()
        st.session_state.record_start_time = time.time()
        st.rerun()

with col2:
    if st.button("⏸️ Pause", use_container_width=True, key="pause"):
        if st.session_state.recorder and st.session_state.recorder.is_recording:
            st.session_state.recorder.pause()
            st.rerun()

with col3:
    if st.button("⏹️ Stop", use_container_width=True, key="stop"):
        if st.session_state.recorder and st.session_state.recorder.is_recording:
            audio_data = st.session_state.recorder.stop()
            st.session_state.record_start_time = None
            
            if audio_data is not None and len(audio_data) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mode_prefix = st.session_state.record_mode
                filename = f"recording_{mode_prefix}_{timestamp}.wav"
                output_path = os.path.join(os.path.dirname(__file__), filename)
                
                wavfile.write(output_path, SAMPLE_RATE, audio_data)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    duration = len(audio_data) / SAMPLE_RATE
                    st.session_state.status_msg = f"✅ Saved: {filename}\n({duration:.1f}s, {file_size/1024:.1f}KB)"
                else:
                    st.session_state.status_msg = "❌ Failed to save"
            else:
                st.session_state.status_msg = "⚠️ No audio recorded"
            
            st.rerun()

# Status display
placeholder = st.empty()

if st.session_state.recorder and st.session_state.recorder.is_recording:
    with placeholder.container():
        if st.session_state.record_start_time:
            elapsed = time.time() - st.session_state.record_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        else:
            time_str = "0s"
        
        if st.session_state.recorder.is_paused:
            st.warning(f"⏸️ Paused ({time_str})")
        else:
            st.info(f"🔴 Recording... ({time_str})")
    
    time.sleep(0.5)
    st.rerun()
else:
    if st.session_state.status_msg:
        st.success(st.session_state.status_msg)
