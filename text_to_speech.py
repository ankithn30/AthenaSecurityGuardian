import pyttsx3
from threading import Thread
import queue

class EdgeTTS:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.audio_queue = queue.Queue()
            
            # List available voices and select one
            voices = self.engine.getProperty('voices')
            print(f"Available voices: {len(voices)}")
            for voice in voices:
                print(f"Voice ID: {voice.id}")
                print(f"Name: {voice.name}")
                print(f"Languages: {voice.languages}")
                print("-" * 20)
            
            if voices:
                # Try to find a female voice for Athena
                female_voice = None
                for voice in voices:
                    if "female" in voice.name.lower():
                        female_voice = voice
                        break
                
                # Set the voice (female if found, otherwise first available)
                self.engine.setProperty('voice', female_voice.id if female_voice else voices[0].id)
            
            # Configure speech properties
            self.engine.setProperty('rate', 150)  # Slightly slower for clarity
            self.engine.setProperty('volume', 1.0)  # Full volume
            
            # Start audio processing thread
            self.audio_thread = Thread(target=self._process_audio_queue, daemon=True)
            self.audio_thread.start()
            
            print("Text-to-speech initialized successfully")
        except Exception as e:
            print(f"Error initializing text-to-speech: {e}")
            raise
    
    def speak(self, text):
        """Add text to speech queue"""
        self.audio_queue.put(text)
    
    def _process_audio_queue(self):
        """Process audio queue in separate thread"""
        while True:
            try:
                text = self.audio_queue.get(timeout=1)
                if text:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.audio_queue.task_done()
            except queue.Empty:
                continue

if __name__ == "__main__":
    print("Initializing Text-to-Speech...")
    tts = EdgeTTS()
    
    test_messages = [
        "Hello! I'm Athena, your security assistant.",
        "Room 203 is on the second floor, east wing.",
        "Emergency services have been notified. Please remain calm.",
        "Visitor parking is free for the first two hours."
    ]
    
    print("\nTesting Text-to-Speech:")
    print("-" * 40)
    for message in test_messages:
        print(f"Speaking: {message}")
        tts.speak(message)
        import time
        time.sleep(4)  # Wait between messages
    
    print("\nTest complete!")