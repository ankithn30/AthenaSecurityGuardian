import threading
import time
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.audio_capture import AudioCapture
from src.speech_to_text import SpeechToText
from src.intent_classifier import IntentClassifier
from src.response_generator import ResponseGenerator
from src.text_to_speech import EdgeTTS
from src.config import Config

class VoiceInteractionSystem:
    def __init__(self):
        print("Initializing Voice Interaction System...")
        self.audio_capture = AudioCapture()
        self.stt = SpeechToText()
        
        # Load trained intent classifier model
        model_path = os.path.join(Config.MODELS_DIR, "intent_classifier", "model")
        self.intent_classifier = IntentClassifier(model_path)
        
        self.response_generator = ResponseGenerator()
        self.tts = EdgeTTS()
        
        self.is_listening = False
        self.wake_word = "athena"  # Wake word for activation
        self.wake_word_confidence_threshold = 0.7  # Confidence threshold for wake word detection
        self.wake_word_alternatives = ["athena", "athina", "atina", "athen"]  # Common mispronunciations

        # Conversation state management
        self.conversation_history = []  # Store recent conversation turns
        self.max_history_length = 10
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3
        }
        self.retry_count = 0
        self.max_retries = 3

    def _detect_wake_word(self, transcription):
        """
        Improved wake word detection with confidence scoring and phonetic matching.

        Args:
            transcription (str): The transcribed text to check

        Returns:
            tuple: (detected, confidence, clean_query)
        """
        if not transcription:
            return False, 0.0, ""

        transcription_lower = transcription.lower().strip()

        # Check for exact matches first (highest confidence)
        if transcription_lower == self.wake_word.lower():
            return True, 1.0, ""

        # Check for wake word at the beginning of transcription
        words = transcription_lower.split()
        first_word = words[0] if words else ""

        # Exact match with first word
        if first_word == self.wake_word.lower():
            remaining = " ".join(words[1:]).strip()
            return True, 0.95, remaining

        # Check alternatives
        best_confidence = 0.0
        best_match = ""
        best_remaining = ""

        for alternative in self.wake_word_alternatives:
            # Check if alternative is at the start
            if transcription_lower.startswith(alternative.lower()):
                confidence = self._calculate_phonetic_similarity(first_word, alternative)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = alternative
                    remaining = transcription_lower[len(alternative):].strip()
                    best_remaining = remaining

        # Check for partial matches within first word
        if best_confidence == 0.0:
            for alternative in self.wake_word_alternatives:
                similarity = self._calculate_phonetic_similarity(first_word, alternative)
                if similarity >= self.wake_word_confidence_threshold:
                    best_confidence = similarity
                    best_match = alternative
                    best_remaining = " ".join(words[1:]).strip()

        if best_confidence >= self.wake_word_confidence_threshold:
            return True, best_confidence, best_remaining

        return False, 0.0, transcription_lower

    def _calculate_phonetic_similarity(self, word1, word2):
        """
        Calculate phonetic similarity between two words using simple metrics.

        Args:
            word1 (str): First word
            word2 (str): Second word (reference)

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if word1 == word2:
            return 1.0

        if len(word1) == 0 or len(word2) == 0:
            return 0.0

        # Length similarity (words of similar length are more likely to be confused)
        len_diff = abs(len(word1) - len(word2))
        len_similarity = max(0, 1.0 - (len_diff / max(len(word1), len(word2))))

        # Character overlap (Levenshtein-like distance)
        # Simple character intersection
        word1_chars = set(word1.lower())
        word2_chars = set(word2.lower())
        char_intersection = len(word1_chars.intersection(word2_chars))
        char_union = len(word1_chars.union(word2_chars))

        if char_union == 0:
            return 0.0

        char_similarity = char_intersection / char_union

        # Position-based similarity (beginning characters are more important)
        min_len = min(len(word1), len(word2))
        if min_len == 0:
            return 0.0

        positional_matches = 0
        for i in range(min_len):
            if i < len(word1) and i < len(word2):
                if word1[i] == word2[i]:
                    # Weight beginning characters more heavily
                    weight = 1.0 + (0.5 * (min_len - i) / min_len)
                    positional_matches += weight
                elif word1[i] in word2 or word2[i] in word1:
                    # Partial credit for similar-sounding characters
                    weight = 0.5 + (0.25 * (min_len - i) / min_len)
                    positional_matches += weight * 0.5

        max_possible = sum([1.0 + (0.5 * (min_len - i) / min_len) for i in range(min_len)])
        positional_similarity = positional_matches / max_possible if max_possible > 0 else 0.0

        # Combine similarities with weights
        overall_similarity = (0.3 * len_similarity +
                            0.3 * char_similarity +
                            0.4 * positional_similarity)

        return min(overall_similarity, 1.0)

    def start_listening(self):
        """Start continuous listening"""
        self.is_listening = True
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        print("Athena is now listening...")
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
    
    def _listen_loop(self):
        """Main listening loop"""
        is_active = False  # Track if we're in an active conversation
        active_timeout = 0  # Timeout counter for active conversation
        
        while self.is_listening:
            try:
                # Visual feedback for listening state
                if not is_active:
                    print("\nListening for wake word 'Athena'...")
                else:
                    print("\nListening for command... (say 'goodbye' to end conversation)")
                
                # Start recording with VAD (Voice Activity Detection)
                print("🎤 Listening...", end="", flush=True)
                audio_data = self.audio_capture.capture_audio_vad(max_duration=8, silence_threshold=2.0)
                print(" ✓", flush=True)

                if audio_data is None or len(audio_data) == 0:
                    print("🔇 No audio detected", flush=True)
                    time.sleep(0.5)
                    continue

                print("🔍 Processing...", flush=True)
                
                if audio_data is not None and len(audio_data) > 0:
                    # Speech to text (using Whisper for better accuracy)
                    transcription = self.stt.transcribe_with_whisper(audio_data)
                    
                    if transcription and len(transcription.strip()) > 0:
                        print(f"👂 Heard: {transcription}")
                        
                        # Check for goodbye command when active
                        if is_active and "goodbye" in transcription.lower():
                            self.tts.speak("Goodbye! Let me know if you need anything else.")
                            is_active = False
                            print("Conversation ended.")
                            continue
                        
                        # Check for wake word if not active
                        if not is_active:
                            wake_detected, confidence, clean_query = self._detect_wake_word(transcription)
                            if wake_detected:
                                is_active = True
                                active_timeout = 0
                                print(f"🎯 Wake word detected! (confidence: {confidence:.2f})")
                                if clean_query:  # If there's more after the wake word
                                    self._process_query(clean_query)
                                else:
                                    self.tts.speak("Hello! How can I assist you today?")
                        else:
                            # Process the query directly when active
                            self._process_query(transcription)
                            active_timeout = 0  # Reset timeout on successful query
                    
                    elif is_active:
                        active_timeout += 1
                        if active_timeout >= 3:  # If no speech detected for 3 cycles
                            is_active = False
                            active_timeout = 0
                            print("Conversation ended due to inactivity")
                
                # Smaller delay when active to be more responsive
                time.sleep(0.2 if is_active else 0.5)
                
            except Exception as e:
                print(f"Error in listening loop: {e}")
                time.sleep(1)
    
    def _process_query(self, transcription):
        """Process a voice query with error recovery and retry logic"""
        try:
            # Clean up transcription
            transcription = transcription.strip()
            if not transcription:
                return

            print("\n🤔 Processing your request...")

            # Classify intent
            intent = self.intent_classifier.classify_intent(transcription)
            print(f"🎯 Detected Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")

            # Enhanced confidence handling
            confidence = intent['confidence']

            if confidence < self.confidence_thresholds['low']:
                # Very low confidence - ask for clarification
                self._handle_low_confidence(transcription)
                return
            elif confidence < self.confidence_thresholds['medium']:
                # Medium confidence - proceed but add uncertainty marker
                intent['intent'] = 'unknown'
                print("⚠️ Medium confidence - treating as unknown intent")

            # Generate response
            print("💭 Generating response...")
            response = self.response_generator.generate_response(intent, transcription)
            print(f"🗣️ Response: {response}")

            # Handle empty responses
            if not response or response.strip() == "":
                self._handle_empty_response(transcription)
                return

            # Add conversational context from history
            response = self._add_conversational_context(response, intent)

            # Store in conversation history
            self._add_to_history(transcription, intent['intent'], confidence, response)

            # Speak response
            print("🔊 Speaking response...")
            self.tts.speak(response)

            # Reset retry count on successful processing
            self.retry_count = 0

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            self._handle_processing_error(transcription)

    def _handle_low_confidence(self, transcription):
        """Handle low confidence situations"""
        self.retry_count += 1

        if self.retry_count <= self.max_retries:
            clarification_responses = [
                "I'm not quite sure I understood that. Could you please repeat or rephrase your question?",
                "I didn't catch that clearly. Could you try saying it again, perhaps a bit louder or clearer?",
                "I'm having trouble understanding. Could you please speak a bit more clearly or rephrase your request?"
            ]

            response = clarification_responses[min(self.retry_count - 1, len(clarification_responses) - 1)]
            print(f"🔊 Asking for clarification (attempt {self.retry_count}/{self.max_retries})")
            self.tts.speak(response)
        else:
            # Max retries reached
            self.retry_count = 0
            response = "I'm still having trouble understanding. Let me help you with some common questions. You can ask me about locations, equipment, or say 'help' for assistance."
            print("🔊 Max retries reached - providing help")
            self.tts.speak(response)

    def _handle_empty_response(self, transcription):
        """Handle empty responses from the system"""
        self.retry_count += 1

        if self.retry_count <= self.max_retries:
            fallback_responses = [
                "I'm not sure I understood that correctly. Could you please rephrase your question?",
                "I couldn't find information about that. Could you try asking in a different way?",
                "I'm having trouble with that request. Could you please try rephrasing it?"
            ]

            response = fallback_responses[min(self.retry_count - 1, len(fallback_responses) - 1)]
            print(f"🔊 Empty response - asking for clarification (attempt {self.retry_count}/{self.max_retries})")
            self.tts.speak(response)
        else:
            self.retry_count = 0
            response = "I'm having persistent trouble with that request. Let me suggest some things I can help you with: locations, equipment, or general information."
            print("🔊 Max retries reached for empty response")
            self.tts.speak(response)

    def _handle_processing_error(self, transcription):
        """Handle processing errors"""
        error_responses = [
            "I apologize, but I encountered an error while processing your request. Could you try asking in a different way?",
            "I'm sorry, something went wrong. Could you please try again?",
            "I had trouble processing that. Could you please rephrase your question?"
        ]

        response = error_responses[0]  # Use first response for errors
        print(f"🔊 Speaking error response: {response}")
        self.tts.speak(response)

        # Reset retry count on error
        self.retry_count = 0

    def _add_conversational_context(self, response, intent):
        """Add conversational context based on intent and history"""
        # Add confidence markers for more natural interaction
        if intent['intent'] == 'greeting':
            return response  # Keep greeting responses as they are
        elif intent['intent'] == 'unknown':
            return "I'm not quite sure about that. " + response
        elif intent['confidence'] > self.confidence_thresholds['high']:
            # Add confidence markers for high-confidence responses
            return "I can help you with that. " + response
        else:
            return response

    def _add_to_history(self, query, intent, confidence, response):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'response': response,
            'timestamp': time.time()
        })

        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

if __name__ == "__main__":
    print("Starting Athena Voice Interaction System...")
    print("-" * 50)
    
    system = VoiceInteractionSystem()
    system.start_listening()
    
    print("\nSystem is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Athena Voice Interaction System...")
        system.is_listening = False
