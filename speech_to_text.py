import speech_recognition as sr
import numpy as np
import os
from src.config import Config
import wave
import io
import tempfile
import torch
import whisper
from typing import Optional

class SpeechToText:
    def __init__(self):
        """Initialize the speech recognition system"""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5

        # Initialize Whisper model
        self.whisper_model = None
        self.use_whisper = True
        self._load_whisper_model()

    def _load_whisper_model(self):
        """Load Whisper model for speech recognition"""
        try:
            # Use base model for good accuracy/speed balance
            model_name = "base"
            print(f"Loading Whisper {model_name} model...")

            # Check if model is already downloaded
            model_path = os.path.join(Config.MODELS_DIR, "whisper")
            if os.path.exists(model_path):
                print("Loading local Whisper model...")
                self.whisper_model = whisper.load_model(model_name, download_root=model_path)
            else:
                print("Downloading Whisper model...")
                self.whisper_model = whisper.load_model(model_name)

            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            print("Falling back to Google Speech Recognition")
            self.use_whisper = False

    def transcribe_with_whisper(self, audio_data) -> str:
        """Transcribe audio using OpenAI Whisper"""
        if not self.use_whisper or self.whisper_model is None:
            return self.transcribe(audio_data)  # Fallback to Google SR

        try:
            # Convert numpy array to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(Config.SAMPLE_RATE)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())

            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path, language="en")
            text = result["text"].strip()

            # Clean up temporary file
            os.unlink(temp_path)

            return text if text else ""

        except Exception as e:
            print(f"Whisper transcription error: {e}")
            # Fallback to Google Speech Recognition
            return self.transcribe(audio_data)

    def transcribe(self, audio_data, language="en-US"):
        """Convert audio to text using Google Speech Recognition"""
        try:
            # Convert numpy array to temporary WAV file
            if isinstance(audio_data, np.ndarray):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    with wave.open(temp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(Config.SAMPLE_RATE)
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        wf.writeframes(audio_int16.tobytes())
            else:
                # If it's raw audio data, save it directly
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    with wave.open(temp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(Config.SAMPLE_RATE)
                        wf.writeframes(audio_data)
            
            # Convert the audio file to AudioFile for recognition
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
            
            # Try to recognize the speech
            try:
                text = self.recognizer.recognize_google(audio, language=language)
            except sr.UnknownValueError:
                text = ""  # Speech was unintelligible
            except sr.RequestError as e:
                print(f"Could not request results from service; {e}")
                text = ""
            except Exception as e:
                if "FLAC" in str(e):
                    print(f"FLAC error: {e}")
                    print("Attempting alternative recognition method...")
                    # Try alternative recognition without Google API
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        print(f"Alternative recognition successful: {text}")
                    except:
                        print("Alternative recognition also failed")
                        text = ""
                else:
                    print(f"Unexpected error: {e}")
                    text = ""
                
            # Clean up temporary file
            os.unlink(temp_path)
            
            return text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def _convert_np_to_audio_data(self, np_array):
        """Convert numpy array to audio data bytes"""
        try:
            # Ensure the audio is float32 and normalized
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)
            
            # Normalize if not already normalized
            if np.max(np.abs(np_array)) > 1.0:
                np_array = np_array / np.max(np.abs(np_array))
            
            # Convert to 16-bit PCM
            audio_16bit = (np_array * 32767).astype(np.int16)
            
            # Create an in-memory wave file
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(Config.SAMPLE_RATE)
                wav_file.writeframes(audio_16bit.tobytes())
            
            return byte_io.getvalue()
            
        except Exception as e:
            print(f"Error converting audio format: {e}")
            return None

# Test script
if __name__ == "__main__":
    stt = SpeechToText()
    
    # Test with a sample audio file if available
    test_file = os.path.join(os.path.dirname(__file__), "..", "tests", "test_recording.wav")
    if os.path.exists(test_file):
        print(f"Testing transcription with {test_file}")
        with wave.open(test_file, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            text = stt.transcribe(frames)
            print(f"Transcribed text: {text}")
    else:
        print("No test file found. Create one using audio_capture.py first.")
