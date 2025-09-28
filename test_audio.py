import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_capture import AudioCapture

def test_audio_capture():
    print("Testing Audio Capture System")
    print("=" * 40)
    
    try:
        # Initialize audio capture
        capture = AudioCapture()
        
        print("Available audio devices:")
        devices = capture.list_audio_devices()
        
        if not devices:
            print("ERROR: No audio devices found!")
            print("Please check:")
            print("  1. Microphone is connected")
            print("  2. Microphone permissions are granted")
            return False
        
        for idx, name in devices:
            print(f"  Device {idx}: {name}")
        
        # Find working device
        print("\nFinding working audio device...")
        working_device = capture.find_working_device()
        
        if working_device is None:
            # Test default device
            success, message = capture.test_audio_device()
            if not success:
                print(f"ERROR: Default device failed - {message}")
                print("Try selecting a specific device manually")
                return False
            print("SUCCESS: Default device works!")
        else:
            print(f"SUCCESS: Found working device ID {working_device}")
        
        # Test audio recording
        print(f"\nTesting audio recording...")
        print("Please speak for 3 seconds when recording starts...")
        input("Press Enter to start recording...")
        
        audio_data = capture.capture_audio(duration=3, device_id=working_device)
        
        if audio_data is not None:
            print("SUCCESS: Audio captured successfully!")
            print(f"  Audio samples: {len(audio_data)}")
            print(f"  Duration: {len(audio_data)/16000:.2f} seconds")
            print(f"  Audio level: {abs(audio_data).max():.3f}")
            
            if abs(audio_data).max() < 0.01:
                print("WARNING: Audio level is very low")
                print("  - Check microphone volume")
                print("  - Speak louder or closer to microphone")
            else:
                print("GOOD: Audio level is adequate")
            
            # Optional: save test recording
            save_audio = input("Save test recording? (y/n): ")
            if save_audio.lower() == 'y':
                try:
                    capture.save_audio(audio_data, "test_recording.wav")
                    print("Audio saved as test_recording.wav")
                except Exception as e:
                    print(f"Failed to save audio: {e}")
            
            print("\n" + "=" * 40)
            print("AUDIO TEST PASSED!")
            print("You can now proceed to Speech-to-Text testing")
            return True
            
        else:
            print("ERROR: Audio capture failed!")
            print("Possible issues:")
            print("  1. Microphone not working")
            print("  2. Permission denied")
            print("  3. Device driver issues")
            return False
            
    except ImportError as e:
        print("ERROR: Required audio library not installed!")
        print(f"Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install sounddevice soundfile")
        return False
        
    except Exception as e:
        print(f"ERROR: Audio test failed with exception: {e}")
        print("\nTroubleshooting steps:")
        print("  1. Check microphone connection")
        print("  2. Grant microphone permissions")
        print("  3. Try: python src/audio_capture.py")
        return False

def test_voice_activity_detection():
    """Test voice activity detection feature"""
    print("\nTesting Voice Activity Detection")
    print("=" * 40)
    
    try:
        capture = AudioCapture()
        
        print("Voice Activity Detection Test:")
        print("  - Speak normally when recording starts")
        print("  - Recording will automatically stop after 2 seconds of silence")
        print("  - Maximum recording time: 15 seconds")
        
        input("Press Enter to start voice detection test...")
        
        audio_data = capture.capture_with_voice_detection(max_duration=15, silence_duration=2)
        
        if audio_data is not None:
            duration = len(audio_data) / 16000
            level = abs(audio_data).max()
            
            print(f"SUCCESS: Voice recording complete!")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Audio level: {level:.3f}")
            
            if duration < 0.5:
                print("WARNING: Very short recording - try speaking longer")
            elif duration > 10:
                print("INFO: Long recording - voice detection may need tuning")
            else:
                print("GOOD: Recording duration looks good")
            
            return True
        else:
            print("ERROR: Voice detection failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Voice detection test failed: {e}")
        return False

def main():
    print("Athena Audio System Test")
    print("=" * 50)
    
    # Basic audio test
    basic_success = test_audio_capture()
    
    if basic_success:
        print("\nBasic audio test PASSED!")
        
        # Ask for advanced test
        advanced = input("\nRun voice activity detection test? (y/n): ")
        if advanced.lower() == 'y':
            voice_success = test_voice_activity_detection()
            if voice_success:
                print("\nAdvanced test PASSED!")
            else:
                print("\nAdvanced test failed (basic audio still works)")
        
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("1. Audio system is working!")
        print("2. Run: python src/speech_to_text.py")
        print("3. Then: python tests/test_stt.py")
        
    else:
        print("\n" + "=" * 50)
        print("AUDIO TEST FAILED!")
        print("Please fix audio issues before proceeding.")
        print("\nTroubleshooting:")
        print("1. Check microphone connection")
        print("2. Grant microphone permissions")
        print("3. Try: python src/audio_capture.py for detailed testing")

if __name__ == "__main__":
    main()