import numpy as np
import pyaudio
from scipy import signal
import wave
import threading
import queue
import time
import os
from src.config import Config
from collections import deque

class AudioCapture:
    def __init__(self, sample_rate=Config.SAMPLE_RATE, chunk_size=Config.CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.recording_thread = None

        # Noise reduction parameters
        self.noise_reduction_enabled = True
        self.noise_profile = None
        self.noise_frames = deque(maxlen=10)  # Store recent frames for noise estimation
        self.vad_threshold = 0.35  # Voice activity detection threshold
        self.noise_alpha = 0.95  # Smoothing factor for noise estimation

    def _apply_noise_reduction(self, audio_chunk):
        """Apply advanced noise reduction using Wiener filtering"""
        if not self.noise_reduction_enabled or self.noise_profile is None:
            return audio_chunk

        try:
            # Compute FFT of the audio chunk
            fft = np.fft.rfft(audio_chunk)
            magnitude = np.abs(fft)
            phase = np.angle(fft)

            # Get noise magnitude
            noise_magnitude = np.abs(self.noise_profile)

            # Apply Wiener filter
            # Wiener filter: H(f) = (P_signal(f) / (P_signal(f) + P_noise(f)))
            signal_power = magnitude ** 2
            noise_power = noise_magnitude ** 2 + 1e-10  # Add small epsilon to avoid division by zero

            # Estimate signal-to-noise ratio
            snr = signal_power / noise_power

            # Wiener filter gain
            wiener_gain = snr / (snr + 1)

            # Apply filter to magnitude
            enhanced_magnitude = magnitude * wiener_gain

            # Ensure we don't over-filter (keep at least 5% of original signal)
            min_magnitude = 0.05 * magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, min_magnitude)

            # Reconstruct the signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.fft.irfft(enhanced_fft)

            return enhanced_audio
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return audio_chunk

    def _update_noise_profile(self, audio_chunk):
        """Update noise profile using current audio chunk"""
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_chunk)
            current_profile = np.abs(fft)

            # Update noise profile using exponential smoothing
            if self.noise_profile is None:
                self.noise_profile = current_profile
            else:
                self.noise_profile = (self.noise_alpha * self.noise_profile +
                                    (1 - self.noise_alpha) * current_profile)
        except Exception as e:
            print(f"Error updating noise profile: {e}")

    def _detect_voice_activity(self, audio_chunk):
        """Simple voice activity detection based on energy threshold"""
        try:
            # Calculate short-term energy
            energy = np.sum(audio_chunk ** 2) / len(audio_chunk)

            # Normalize energy (simple approach)
            normalized_energy = energy / (np.var(audio_chunk) + 1e-10)

            return normalized_energy > self.vad_threshold
        except Exception as e:
            print(f"Error in VAD: {e}")
            return False

    def _preprocess_audio(self, audio_chunk):
        """Apply audio preprocessing including normalization and DC removal"""
        try:
            # Remove DC component
            audio_chunk = audio_chunk - np.mean(audio_chunk)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0.95:  # If close to clipping
                audio_chunk = audio_chunk / max_val * 0.9

            return audio_chunk
        except Exception as e:
            print(f"Error in audio preprocessing: {e}")
            return audio_chunk

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.frames = []
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop recording and return the audio data"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        return np.concatenate(self.frames) if self.frames else None
        
    def _record(self):
        """Internal method to record audio data with noise filtering"""
        # Initialize noise profile with first few frames
        noise_init_frames = 5
        frame_count = 0

        while self.is_recording:
            try:
                # Read raw audio data
                raw_data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)

                # Preprocess audio (DC removal, normalization)
                processed_data = self._preprocess_audio(raw_data)

                # Update noise profile during initialization phase
                if frame_count < noise_init_frames:
                    self._update_noise_profile(processed_data)
                    frame_count += 1
                    # Store original data during noise profile initialization
                    self.frames.append(raw_data)
                else:
                    # Apply noise reduction after initialization
                    if self.noise_reduction_enabled:
                        filtered_data = self._apply_noise_reduction(processed_data)
                        self.frames.append(filtered_data)
                    else:
                        self.frames.append(processed_data)

            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def list_audio_devices(self):
        """List available audio input devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only include input devices
                    devices.append((i, device_info['name']))
            except Exception as e:
                print(f"Error getting device info for index {i}: {str(e)}")
        return devices

    def find_working_device(self, test_duration=1):
        """Find a working audio input device"""
        devices = self.list_audio_devices()
        
        for device_id, device_name in devices:
            print(f"\nTesting device {device_id}: {device_name}")
            try:
                stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=self.chunk_size
                )
                
                # Try to read some data
                data = stream.read(self.chunk_size)
                if len(data) > 0:
                    print("✓ Device is working!")
                    stream.close()
                    return device_id
                    
            except Exception as e:
                print(f"✗ Error testing device: {str(e)}")
                
            if stream:
                stream.close()
        
        return None
        
    def record(self, duration=None, device_id=None):
        """
        Record audio from the microphone for a specified duration.
        If duration is None, records until stop() is called.
        """
        self.frames = []
        self.is_recording = True
        
        # Open stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=self.chunk_size
        )
        
        if duration:
            # Record for specified duration
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                if not self.is_recording:
                    break
                data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)
                self.frames.append(data)
            self.stop()
        else:
            # Record until stop() is called
            while self.is_recording:
                data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)
                self.frames.append(data)
            

    
    def stop(self):
        """
        Stop recording.
        """
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def save_wav(self, filename):
        """
        Save recorded audio to a WAV file.
        """
        if not self.frames:
            return False

        # Convert float32 to int16 for WAV file
        audio_data = np.concatenate(self.frames, axis=0)
        audio_data = np.int16(audio_data * 32767)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return True

    def capture_audio_vad(self, max_duration=10, silence_threshold=3.0):
        """
        Capture audio using voice activity detection.
        Records until speech ends or max_duration is reached.

        Args:
            max_duration (float): Maximum recording duration in seconds
            silence_threshold (float): Seconds of silence before stopping

        Returns:
            numpy array: Recorded audio data
        """
        self.frames = []
        self.is_recording = True
        self.noise_profile = None  # Reset noise profile for new recording

        # Open stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start recording thread with VAD
        self.recording_thread = threading.Thread(target=self._record_vad,
                                               args=(max_duration, silence_threshold))
        self.recording_thread.start()

        # Wait for recording to complete
        self.recording_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        return np.concatenate(self.frames) if self.frames else None

    def _record_vad(self, max_duration, silence_threshold):
        """Record audio with voice activity detection"""
        noise_init_frames = 5
        frame_count = 0
        silence_frames = 0
        max_frames = int(self.sample_rate * max_duration / self.chunk_size)
        total_frames = 0

        while self.is_recording and total_frames < max_frames:
            try:
                # Read raw audio data
                raw_data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)

                # Preprocess audio
                processed_data = self._preprocess_audio(raw_data)

                # Update noise profile during initialization
                if frame_count < noise_init_frames:
                    self._update_noise_profile(processed_data)
                    frame_count += 1
                    self.frames.append(raw_data)
                else:
                    # Apply noise reduction
                    if self.noise_reduction_enabled:
                        filtered_data = self._apply_noise_reduction(processed_data)
                    else:
                        filtered_data = processed_data

                    # Voice activity detection
                    if self._detect_voice_activity(filtered_data):
                        silence_frames = 0  # Reset silence counter
                        self.frames.append(filtered_data)
                    else:
                        silence_frames += 1
                        self.frames.append(filtered_data)  # Keep silence for context

                    # Check if we've had enough silence to stop
                    silence_threshold_frames = int(silence_threshold * self.sample_rate / self.chunk_size)
                    if silence_frames > silence_threshold_frames:
                        break

                total_frames += 1

            except Exception as e:
                print(f"Error in VAD recording: {e}")
                break

        self.is_recording = False

    def __del__(self):
        """
        Clean up PyAudio resources.
        """
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
        except Exception as e:
            # Handle cleanup errors during deletion
            print(f"Error cleaning up audio resources: {str(e)}")
            return False

# Test audio capture
if __name__ == "__main__":
    capture = AudioCapture()
    print("Available devices:", capture.list_audio_devices())
    print("Testing audio capture...")
    audio = capture.capture_audio(3)
    print(f"Captured audio shape: {audio.shape if audio is not None else 'None'}")
