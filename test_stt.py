import sys
import os
import wave
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.audio_capture import AudioCapture
from src.speech_to_text import SpeechToText

def test_speech_to_text():
    print("\n=== Testing Speech-to-Text System ===")
    
    try:
        # Initialize components
        print("\nTest 1: Initializing components...")
        stt = SpeechToText()
        print("✓ Speech recognition system initialized")
        
        # First test: Live audio capture and transcription
        print("\nTest 2: Testing live transcription")
        print("-" * 40)
        
        # Create an audio capture instance
        capture = AudioCapture()
        device_id = capture.find_working_device()
        
        if device_id is None:
            print("✗ No working audio device found!")
            return
            
        print("Recording 5 seconds of audio...")
        print("Please speak something...")
        
        # Record audio
        capture.record(duration=5, device_id=device_id)
        
        # Save the recording
        test_file = "test_recording.wav"
        if capture.save_wav(test_file):
            print("✓ Audio captured and saved successfully")
            
            # Transcribe the recorded audio
            print("\nTranscribing recorded audio...")
            with wave.open(test_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                text = stt.transcribe(frames)
                
                print("\nLive Recording Transcription Results:")
                print("-" * 40)
                print(text if text else "(No speech detected)")
                print("-" * 40)
                
                if text:
                    print("✓ Successfully transcribed audio")
                else:
                    print("⚠ Warning: No text was transcribed")
        else:
            print("✗ Failed to save audio recording")
        
        # Test with numpy array input
        print("\nTest 3: Testing with numpy array input...")
        try:
            # Read the WAV file into a numpy array
            with wave.open(test_file, 'rb') as wf:
                # Read the frames and convert to numpy array
                frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                
                print("Transcribing numpy array...")
                text = stt.transcribe(audio_array)
                
                print("\nNumpy Array Transcription Results:")
                print("-" * 40)
                print(text if text else "(No speech detected)")
                print("-" * 40)
                
                if text:
                    print("✓ Successfully transcribed numpy array")
                else:
                    print("⚠ Warning: No text was transcribed from numpy array")
                    
        except Exception as e:
            print(f"✗ Error during transcription: {str(e)}")
            
    except Exception as e:
        print(f"✗ Test error: {str(e)}")

if __name__ == "__main__":
    test_speech_to_text()
