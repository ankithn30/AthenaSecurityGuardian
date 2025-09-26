import whisper
import pyaudio
import numpy as np
import threading
import queue
import time
import os
import ssl
from urllib.request import urlopen
import cv2
import subprocess

class VoiceAgent:
    def __init__(self, model_size="base", language="en"):
        """
        Initialize the Voice Agent with Whisper model.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language (str): Language code for transcription
        """
        print("Loading Whisper model...")
        # Fix SSL certificate verification for corporate networks/self-signed certs
        try:
            # Create unverified SSL context
            ssl_context = ssl._create_unverified_context()
            
            # Monkey patch urllib.request to use unverified context
            original_urlopen = urlopen
            def unverified_urlopen(url, data=None, timeout=None, *, cafile=None, capath=None, cadefault=False, context=None):
                return original_urlopen(url, data=data, timeout=timeout, cafile=cafile, capath=capath, cadefault=cadefault, context=ssl_context)
            
            import urllib.request
            urllib.request.urlopen = unverified_urlopen
            
            # Also set environment variable for HF Hub
            os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
            print("SSL verification disabled for model download.")
            
            self.model = whisper.load_model(model_size)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Troubleshooting steps:")
            print("1. Run: export HF_HUB_DISABLE_SSL_VERIFY=1")
            print("2. Or download model manually: git clone https://huggingface.co/openai/whisper-base ~/.cache/whisper")
            print("3. Then retry.")
            raise
        self.language = language
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.commands = {
            "start detection": "Start computer vision detection",
            "stop detection": "Stop computer vision detection",
            "take photo": "Capture a photo",
            "run benchmark": "Run performance benchmark",
            "help": "Show available commands"
        }
        print("Voice Agent initialized. Available commands:", list(self.commands.keys()))

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback to capture audio data."""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start_listening(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=16000, record_seconds=5):
        """
        Start continuous listening for voice commands.
        
        Args:
            chunk (int): Audio chunk size
            format: PyAudio format
            channels (int): Number of audio channels
            rate (int): Sampling rate
            record_seconds (int): Duration to record for each transcription attempt
        """
        self.is_listening = True
        p = pyaudio.PyAudio()

        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk,
                        stream_callback=self.audio_callback)

        print("Listening for voice commands... Speak now!")
        stream.start_stream()

        try:
            while self.is_listening:
                # Collect audio for record_seconds
                audio_data = []
                start_time = time.time()
                while time.time() - start_time < record_seconds and self.is_listening:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        audio_data.append(np.frombuffer(data, dtype=np.int16))
                    except queue.Empty:
                        continue

                if audio_data:
                    # Concatenate and convert to float32
                    audio_array = np.concatenate(audio_data)
                    audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

                    # Transcribe with Whisper
                    result = self.model.transcribe(audio_array, language=self.language)
                    text = result["text"].strip().lower()

                    if text:
                        print(f"Transcribed: '{text}'")
                        self.process_command(text)

                time.sleep(0.5)  # Brief pause between recordings

        except KeyboardInterrupt:
            print("Stopping voice agent...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.is_listening = False

    def process_command(self, text):
        """Process transcribed text for voice commands."""
        for command, action in self.commands.items():
            if command in text:
                print(f"Command recognized: {action}")
                if "start detection" in text:
                    subprocess.Popen(["python", "src/yolo_webcam.py"])
                    print("Starting webcam detection...")
                elif "stop detection" in text:
                    subprocess.run(["pkill", "-f", "yolo_webcam.py"])
                    print("Stopping webcam detection...")
                elif "take photo" in text:
                    # Camera selection logic (same as yolo_webcam.py)
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        def list_cameras():
                            index = 0
                            arr = []
                            while True:
                                cap_test = cv2.VideoCapture(index)
                                if not cap_test.read()[0]:
                                    break
                                arr.append(index)
                                cap_test.release()
                                index += 1
                            return arr
                        
                        available_cameras = list_cameras()
                        if available_cameras:
                            cap = cv2.VideoCapture(available_cameras[0])
                            print(f"Using camera at index {available_cameras[0]}")
                        else:
                            print("No camera available for photo capture")
                            return
                    
                    ret, frame = cap.read()
                    if ret:
                        photo_path = "../data/photo.jpg"
                        cv2.imwrite(photo_path, frame)
                        print(f"Photo captured and saved to {photo_path}")
                    else:
                        print("Failed to capture photo")
                    cap.release()
                elif "run benchmark" in text:
                    subprocess.Popen(["python", "src/benchmark_yolo.py"])
                    print("Running benchmark...")
                elif "help" in text:
                    print("Available commands: " + ", ".join(self.commands.keys()))
                return
        print("No matching command found. Available: " + ", ".join(self.commands.keys()))

    def transcribe_file(self, audio_file):
        """
        Transcribe an audio file.
        
        Args:
            audio_file (str): Path to audio file
        """
        if not os.path.exists(audio_file):
            print(f"Audio file {audio_file} not found.")
            return None
        
        result = self.model.transcribe(audio_file, language=self.language)
        text = result["text"].strip()
        print(f"Transcribed from file: '{text}'")
        self.process_command(text)
        return text

    def stop_listening(self):
        """Stop the voice agent listening."""
        self.is_listening = False

# Example usage
if __name__ == "__main__":
    agent = VoiceAgent(model_size="base")
    
    # Option 1: Transcribe from file
    # agent.transcribe_file("path/to/audio.wav")
    
    # Option 2: Start real-time listening
    try:
        agent.start_listening()
    except KeyboardInterrupt:
        agent.stop_listening()
        print("Voice agent stopped.")
