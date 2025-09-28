import sys
import os
import time
import wave

# Add the project root to the Python path so we can import our module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.audio_capture import AudioCapture

def test_audio_capture():
    print("\n=== Testing AudioCapture ===")
    
    # Create AudioCapture instance
    audio = AudioCapture()
    
    # Test 1: List audio devices
    print("\nTest 1: Listing audio devices...")
    devices = audio.list_audio_devices()
    print(f"Found {len(devices)} audio input devices:")
    for device_id, device_name in devices:
        print(f"  Device {device_id}: {device_name}")
    
    # Test 2: Find working device
    print("\nTest 2: Finding working device...")
    device_id = audio.find_working_device()
    if device_id is not None:
        print(f"Found working device with ID: {device_id}")
    else:
        print("No working audio device found!")
        return
    
    # Test 3: Record audio
    duration = 3  # Record for 3 seconds
    print(f"\nTest 3: Recording {duration} seconds of audio...")
    audio.record(duration=duration, device_id=device_id)
    print("Recording complete!")
    
    # Test 4: Save audio
    output_file = "test_recording.wav"
    print(f"\nTest 4: Saving audio to {output_file}...")
    if audio.save_wav(output_file):
        print("Audio saved successfully!")
        
        # Verify the WAV file
        if os.path.exists(output_file):
            try:
                with wave.open(output_file, 'rb') as wf:
                    print("\nWAV file details:")
                    print(f"  Number of channels: {wf.getnchannels()}")
                    print(f"  Sample width: {wf.getsampwidth()} bytes")
                    print(f"  Frame rate: {wf.getframerate()} Hz")
                    print(f"  Number of frames: {wf.getnframes()}")
                    print(f"  File size: {os.path.getsize(output_file)} bytes")
            except Exception as e:
                print(f"Error reading WAV file: {str(e)}")
        else:
            print("Error: WAV file was not created!")
    else:
        print("Error: Failed to save audio!")

if __name__ == "__main__":
    test_audio_capture()