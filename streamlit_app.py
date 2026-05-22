import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import tempfile
import os
from pathlib import Path

# Function to check which models are cached locally
def get_cached_models():
    """Check which Whisper models are already downloaded locally"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cached_models = []
    
    model_map = {
        "tiny": "models--openai--whisper-tiny",
        "base": "models--openai--whisper-base",
        "small": "models--openai--whisper-small",
        "medium": "models--openai--whisper-medium",
        "large-v3": "models--openai--whisper-large-v3"
    }
    
    for model_name, dir_name in model_map.items():
        if (cache_dir / dir_name).exists():
            cached_models.append(model_name)
    
    # If no models cached yet, return all (they'll download on first use)
    # But prefer returning cached ones
    return cached_models if cached_models else ["tiny", "base", "small"]

# Page config
st.set_page_config(
    page_title="System Audio Transcriber",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 System Audio Transcriber")
st.markdown("Powered by faster-whisper (local · free · private)")

# Initialize session state
if "recording" not in st.session_state:
    st.session_state.recording = False
    st.session_state.audio_data = None
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.sample_rate = 16000

# Get available devices - keep only Speaker and Microphone
def get_working_devices():
    """Get only Speaker (Stereo Mix) and one working Microphone"""
    devices = sd.query_devices()
    devices_list = []
    
    speaker_added = False
    mic_added = False
    
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].lower()
            
            # Add Stereo Mix as Speaker option (only once)
            if 'stereo mix' in name and not speaker_added:
                devices_list.append((i, f"🔊 Speaker - {d['name']}"))
                speaker_added = True
            
            # Add first working Microphone (only once)
            elif ('microphone' in name or 'mic' in name) and not mic_added and 'array' in name:
                devices_list.append((i, f"🎤 Microphone - {d['name']}"))
                mic_added = True
    
    return devices_list if devices_list else []

working_devices = get_working_devices()
device_options = [dev[1] for dev in working_devices]

# Function to get device's supported sample rate
def get_device_sample_rate(device_index):
    try:
        device_info = sd.query_devices(device_index)
        # Try common sample rates in order
        for rate in [48000, 44100, 32000, 16000, 8000]:
            try:
                sd.check_input_settings(device=device_index, samplerate=rate)
                return rate
            except:
                continue
        return 16000  # fallback
    except:
        return 16000

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Settings")
    
    # Device selection - only show Speaker and Microphone
    if device_options:
        selected_device = st.selectbox(
            "Select Input Device",
            options=device_options,
            index=0
        )
        # Find the device index from working_devices
        device_index = None
        for dev_id, dev_name in working_devices:
            if dev_name == selected_device:
                device_index = dev_id
                break
    else:
        st.error("❌ No working input devices found!")
        device_index = None
    
    # Get available cached models
    cached_models = get_cached_models()
    default_index = cached_models.index("base") if "base" in cached_models else 0
    
    # Model selection
    model_size = st.radio(
        "Whisper Model Size",
        options=cached_models,
        index=default_index,
        help="Showing only cached/downloaded models. Others will download on first use."
    )
    
    # Show loading time estimate
    load_times = {
        "tiny": "~1-2 sec",
        "base": "~2-3 sec",
        "small": "~5-10 sec",
        "medium": "~20-30 sec",
        "large-v3": "~60+ sec"
    }
    st.caption(f"⏱️ Load time: {load_times[model_size]}")

with col2:
    st.subheader("📊 Status")
    
    # Show cached models
    st.info(f"✅ **Cached Models:** {', '.join(cached_models)}")
    
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

# Recording controls
st.subheader("🔴 Recording Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔴 Start Recording", key="record_btn", use_container_width=True):
        if device_index is None:
            status_placeholder.error("❌ No input device selected!")
        else:
            st.session_state.recording = True
            st.session_state.recording_start_time = None
            
            try:
                # Unlimited recording with very large buffer
                max_samples = 16000 * 3600  # ~1 hour at 16kHz (will adapt to actual sample rate)
                
                audio_data = None
                recorded_rate = 16000
                
                # Try different sample rates and settings
                sample_rates = [16000, 8000, 32000, 44100]
                dtypes = [np.int16, np.float32]
                channels_list = [1, 2]
                
                for rate in sample_rates:
                    if audio_data is not None:
                        break
                    for dtype in dtypes:
                        if audio_data is not None:
                            break
                        for ch in channels_list:
                            try:
                                # Record with very large duration (until user stops)
                                import time
                                audio_list = []
                                start_time = time.time()
                                
                                # Record in chunks so UI can respond
                                chunk_duration = 0.5  # 500ms chunks
                                chunk_size = int(rate * chunk_duration)
                                
                                while st.session_state.recording:
                                    chunk = sd.rec(chunk_size, samplerate=rate, channels=ch, device=device_index, dtype=dtype, blocksize=0)
                                    sd.wait()
                                    audio_list.append(chunk)
                                    
                                    # Update timer
                                    elapsed = time.time() - start_time
                                    mins = int(elapsed // 60)
                                    secs = int(elapsed % 60)
                                    status_placeholder.info(f"🔴 Recording... {mins}m {secs}s")
                                
                                if audio_list:
                                    audio_data = np.concatenate(audio_list)
                                    recorded_rate = rate
                                
                                break
                            except:
                                continue
                
                if audio_data is not None and len(audio_data) > 0:
                    st.session_state.audio_data = audio_data
                    st.session_state.sample_rate = recorded_rate
                    st.session_state.recording = False
                    
                    elapsed = time.time() - start_time
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    status_placeholder.success(f"✅ Recording complete! {mins}m {secs}s ({recorded_rate} Hz)")
                else:
                    st.session_state.recording = False
                    status_placeholder.error("❌ Device doesn't support recording - try a different device")
                    
            except Exception as e:
                status_placeholder.error(f"❌ Error recording: {str(e)}")
                st.session_state.recording = False

with col2:
    if st.button("⏹️ Stop Recording", key="stop_btn", use_container_width=True):
        st.session_state.recording = False
        status_placeholder.warning("Recording stopped")

with col3:
    if st.button("🗑️ Clear Recording", key="clear_btn", use_container_width=True):
        st.session_state.audio_data = None
        st.session_state.recording = False
        status_placeholder.info("Recording cleared")

# Audio Playback
if st.session_state.audio_data is not None:
    st.subheader("🔊 Playback")
    
    # Convert audio to bytes for playback
    import io
    from scipy.io import wavfile
    
    audio = st.session_state.audio_data.flatten().astype(np.float32)
    
    # Create WAV file in memory
    audio_buffer = io.BytesIO()
    wavfile.write(audio_buffer, st.session_state.sample_rate, (audio * 32767).astype(np.int16))
    audio_buffer.seek(0)
    
    st.audio(audio_buffer, format="audio/wav", sample_rate=st.session_state.sample_rate)

# Transcription
if st.session_state.audio_data is not None:
    st.subheader("📝 Transcription")
    
    if st.button("🚀 Transcribe Audio", use_container_width=True):
        progress_placeholder.info(f"Loading {model_size} model...")
        
        try:
            # Load model
            if st.session_state.model is None or st.session_state.model_size != model_size:
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                st.session_state.model = model
                st.session_state.model_size = model_size
            else:
                model = st.session_state.model
            
            progress_placeholder.info("Transcribing audio...")
            
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import scipy.io.wavfile as wavfile
                from scipy import signal
                
                # Get the actual recorded sample rate
                recorded_rate = st.session_state.sample_rate
                audio = st.session_state.audio_data.flatten()
                
                # Normalize audio to prevent clipping
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
                
                # Resample to 16kHz for Whisper if needed
                if recorded_rate != 16000:
                    num_samples = int(len(audio) * 16000 / recorded_rate)
                    audio = signal.resample(audio, num_samples)
                    target_rate = 16000
                else:
                    target_rate = 16000
                
                # Convert to int16 for WAV file
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(tmp.name, target_rate, audio_int16)
                temp_path = tmp.name
            
            # Transcribe (auto-detect language)
            segments, info = model.transcribe(temp_path)
            
            # Display detected language
            detected_language = info.language if hasattr(info, 'language') else "Unknown"
            
            # Collect transcription
            full_text = " ".join([segment.text for segment in segments])
            
            # Clean up
            os.remove(temp_path)
            
            # Display results
            progress_placeholder.success("✅ Transcription complete!")
            
            # Show detected language
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                st.info(f"🌍 **Detected Language:** {detected_language.upper() if detected_language != 'Unknown' else 'Unknown'}")
            
            st.text_area(
                "Transcribed Text",
                value=full_text,
                height=150,
                disabled=True
            )
            
            # Display segments
            with st.expander("📋 Detailed Segments"):
                for segment in segments:
                    st.write(f"**[{segment.start:.2f}s - {segment.end:.2f}s]** {segment.text}")
            
        except Exception as e:
            progress_placeholder.error(f"❌ Error transcribing: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
