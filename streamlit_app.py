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
    st.session_state.recording_thread = None
    st.session_state.recording_start_time = None
    st.session_state.device_index = None  # Add this line

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
            
            # Add first working Microphone - prefer ones with 'array', but fallback to any mic
            elif ('microphone' in name or 'mic' in name) and not mic_added:
                # Prefer devices with 'array' or 'smart' in the name
                if 'array' in name or 'smart' in name or 'mapper' in name:
                    devices_list.append((i, f"🎤 Microphone - {d['name']}"))
                    mic_added = True
    
    return devices_list if devices_list else []

working_devices = get_working_devices()
device_options = [dev[1] for dev in working_devices]

# Function to record audio in background thread
def record_audio_background(device_index):
    """Record audio in background using threading"""
    import sys
    
    print(f"[RECORD] Starting recording on device {device_index}", file=sys.stderr)
    
    try:
        SAMPLE_RATE = 16000
        CHANNELS = 1
        CHUNK_SIZE = 8000  # 0.5 seconds at 16kHz
        
        audio_chunks = []
        chunk_count = 0
        
        # Record chunks while recording flag is True
        while st.session_state.recording:
            try:
                print(f"[RECORD] Recording chunk {chunk_count}...", file=sys.stderr)
                chunk = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=CHANNELS,
                             device=device_index, dtype=np.int16, blocksize=0)
                sd.wait()
                audio_chunks.append(chunk)
                chunk_count += 1
                print(f"[RECORD] Chunk {chunk_count} captured: {len(chunk)} samples", file=sys.stderr)
            except Exception as e:
                print(f"[RECORD] Error recording chunk: {e}", file=sys.stderr)
                break
        
        # Combine all chunks into one array
        print(f"[RECORD] Recording stopped. Total chunks: {len(audio_chunks)}", file=sys.stderr)
        
        if audio_chunks and len(audio_chunks) > 0:
            combined_audio = np.concatenate(audio_chunks)
            print(f"[RECORD] Combined audio: {len(combined_audio)} samples", file=sys.stderr)
            st.session_state.audio_data = combined_audio
            st.session_state.sample_rate = SAMPLE_RATE
            print(f"[RECORD] Audio saved to session state", file=sys.stderr)
        else:
            print(f"[RECORD] No audio chunks captured!", file=sys.stderr)
            st.session_state.audio_data = None
            
    except Exception as e:
        print(f"[RECORD] Exception: {e}", file=sys.stderr)
        st.session_state.audio_data = None

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
        for dev_id, dev_name in working_devices:
            if dev_name == selected_device:
                st.session_state.device_index = dev_id
                break
        
        # DEBUG: Show selected device
        st.caption(f"Using Device {st.session_state.device_index}: {selected_device}")
    else:
        st.error("❌ No working input devices found!")
        st.session_state.device_index = None
    
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
    
    # Test microphone button
    if st.button("🔧 Test Microphone", key="test_mic_btn", use_container_width=True):
        if st.session_state.device_index is not None:
            st.info("Testing microphone... please wait 1 second...")
            try:
                test_audio = sd.rec(16000, samplerate=16000, channels=1, 
                                   device=st.session_state.device_index, dtype=np.int16, blocksize=0)
                sd.wait()
                max_level = np.max(np.abs(test_audio))
                if max_level > 0:
                    st.success(f"Microphone works! (Level: {max_level})")
                else:
                    st.warning(f"No sound detected (Level: 0)")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("📊 Status")
    
    # Show cached models
    st.info(f"✅ **Cached Models:** {', '.join(cached_models)}")
    
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

# Recording controls
st.subheader("🔴 Recording Controls")

col1, col2, col3 = st.columns(3)
timer_placeholder = st.empty()  # For recording timer

with col1:
    if st.button("🔴 Start Recording", key="record_btn", use_container_width=True):
        if st.session_state.device_index is None:
            status_placeholder.error("No input device selected!")
        else:
            # Reset audio
            st.session_state.audio_data = None
            st.session_state.recording = True
            st.session_state.recording_start_time = None
            
            print(f"\n[START] User clicked Start Recording with device {st.session_state.device_index}", file=__import__('sys').stderr)
            
            # Start recording in background thread
            thread = threading.Thread(target=record_audio_background, args=(st.session_state.device_index,), daemon=True)
            st.session_state.recording_thread = thread
            thread.start()
            
            print(f"[START] Recording thread started", file=__import__('sys').stderr)
            
            status_placeholder.info("Recording... Speak now!")

with col2:
    if st.button("⏹️ Stop Recording", key="stop_btn", use_container_width=True):
        if st.session_state.recording:
            print(f"\n[STOP] User clicked Stop Recording", file=__import__('sys').stderr)
            st.session_state.recording = False
            print(f"[STOP] Recording flag set to False", file=__import__('sys').stderr)
            
            # Wait for thread to finish
            if st.session_state.recording_thread:
                print(f"[STOP] Waiting for thread to finish...", file=__import__('sys').stderr)
                st.session_state.recording_thread.join(timeout=3)
                print(f"[STOP] Thread finished", file=__import__('sys').stderr)
            
            # Show result
            if st.session_state.audio_data is not None and len(st.session_state.audio_data) > 0:
                samples = len(st.session_state.audio_data)
                st.success(f"Recording complete! ({samples} samples)")
            else:
                st.error("No audio captured - try again")
        else:
            st.info("No recording in progress")

with col3:
    if st.button("🗑️ Clear Recording", key="clear_btn", use_container_width=True):
        st.session_state.audio_data = None
        st.session_state.recording = False
        status_placeholder.info("Recording cleared")

# Show recording timer while recording
if st.session_state.recording:
    import time
    if st.session_state.recording_start_time is None:
        st.session_state.recording_start_time = time.time()
    
    elapsed = time.time() - st.session_state.recording_start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    timer_placeholder.info(f"⏱️ **Recording: {mins}m {secs}s**")
    st.rerun()

# Check if we have audio data
has_audio = False
if st.session_state.audio_data is not None:
    try:
        has_audio = len(st.session_state.audio_data) > 0
    except:
        has_audio = False

if not has_audio:
    st.warning("⏳ No audio recorded yet. Click 'Start Recording' to begin.")

# Audio Playback
if has_audio:
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
if has_audio:
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
