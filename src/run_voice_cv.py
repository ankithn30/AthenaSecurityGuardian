import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_agent import VoiceAgent
import subprocess
import time

def main():
    print("Starting Voice-Controlled CV System...")
    agent = VoiceAgent(model_size="base")
    
    print("Voice agent ready. Say commands like 'start detection' or 'run benchmark'.")
    
    # Start listening in a separate thread to allow integration
    import threading
    listening_thread = threading.Thread(target=agent.start_listening)
    listening_thread.daemon = True
    listening_thread.start()
    
    # Keep main thread alive for integration
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop_listening()
        print("System stopped.")

if __name__ == "__main__":
    main()
