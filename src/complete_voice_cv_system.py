import sys
import os
import cv2
import numpy as np
import time
import logging
import json
import threading
from typing import Tuple, List, Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteVoiceCVSystem:
    def __init__(self, database_dir: str = None, confidence_threshold: float = 0.6):
        """
        Initialize the complete voice-controlled computer vision system.

        Args:
            database_dir: Directory containing the face database (if None, auto-detect from script location)
            confidence_threshold: Minimum confidence level to trigger welcome message
        """
        # Auto-detect database location if not provided
        if database_dir is None:
            # Get the directory two levels up from this script, then look for known_faces
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            database_dir = os.path.join(project_root, "known_faces")

        self.database_dir = database_dir
        self.confidence_threshold = confidence_threshold
        self.database_file = os.path.join(database_dir, "faces_metadata.json")
        self.known_faces = {}
        self.face_cascade = None
        self.camera = None
        self.is_running = False
        self.athena = None  # Athena voice system

        # Initialize face detection
        self.initialize_face_detection()

        # Load the database
        self.load_database()

        # Initialize Athena
        self.initialize_athena()

    def initialize_athena(self):
        """Initialize the Athena voice interaction system."""
        try:
            # Import from the voice_assistant directory (go up one level from src/)
            voice_assistant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'voice_assistant')
            sys.path.insert(0, voice_assistant_path)
            print(f"Adding path: {voice_assistant_path}")
            from voice import VoiceInteractionSystem
            print("Creating VoiceInteractionSystem...")
            self.athena = VoiceInteractionSystem()
            print("Starting listening...")
            self.athena.start_listening()
            logger.info("Athena voice system initialized and started")
        except Exception as e:
            logger.error(f"Failed to initialize Athena: {e}")
            import traceback
            traceback.print_exc()
            self.athena = None

    def initialize_face_detection(self):
        """Initialize the face detection cascade classifier."""
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ]

        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    logger.info(f"Loaded Haar Cascade from: {path}")
                    break

        if self.face_cascade is None or self.face_cascade.empty():
            error_msg = "Could not load Haar Cascade file. Make sure OpenCV is properly installed."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def load_database(self):
        """Load the face database from file."""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    self.known_faces = json.load(f)
                logger.info(f"Loaded database with {len(self.known_faces)} known faces")
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                self.known_faces = {}
        else:
            logger.info("No database found, starting with empty database")
            self.known_faces = {}

    def identify_faces(self, image: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Identify faces in the input image by comparing with known faces.

        Args:
            image: Input image in BGR format

        Returns:
            List of (name, face_id, similarity_score) tuples for each detected face
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            identifications = []

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                # Compare with known faces
                best_match = None
                best_score = 0.0
                best_face_id = None

                for face_id, metadata in self.known_faces.items():
                    known_face_path = os.path.join(self.database_dir, metadata['image_path'])

                    if os.path.exists(known_face_path):
                        known_face = cv2.imread(known_face_path, cv2.IMREAD_GRAYSCALE)

                        if known_face is not None and known_face.shape[0] > 0 and known_face.shape[1] > 0:
                            # Resize faces to same size for comparison
                            try:
                                resized_current = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))

                                # Calculate similarity using histogram comparison
                                hist_current = cv2.calcHist([resized_current], [0], None, [256], [0, 256])
                                hist_known = cv2.calcHist([known_face], [0], None, [256], [0, 256])
                                similarity = cv2.compareHist(hist_current, hist_known, cv2.HISTCMP_CORREL)

                                if similarity > best_score and similarity > 0.4:
                                    best_score = similarity
                                    best_match = metadata['name']
                                    best_face_id = face_id
                                    logger.debug(f"Face match found: {metadata['name']} with score {similarity:.3f}")

                            except Exception as e:
                                logger.debug(f"Error comparing faces: {e}")
                                continue

                if best_match:
                    identifications.append((best_match, best_face_id, best_score))
                    logger.info(f"Face identified as: {best_match} (score: {best_score:.3f})")
                else:
                    identifications.append(("Unknown", "unknown", 0.0))
                    logger.debug(f"Face not identified - best score: {best_score:.3f}")

            return identifications

        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return []

    def check_confidence_and_respond(self, recognitions: List[Tuple[str, str, float]]):
        """
        Check recognition results and activate Athena for high confidence matches or unknown persons.
        Uses hysteresis to prevent flip-flopping between known/unknown states.

        Args:
            recognitions: List of (name, face_id, confidence) tuples
        """
        current_time = time.time()

        # Initialize hysteresis tracking if not exists
        if not hasattr(self, 'last_known_state'):
            self.last_known_state = None
            self.last_known_confidence = 0.0
            self.state_change_cooldown = 0

        # Check for unknown persons first (higher priority)
        for name, face_id, confidence in recognitions:
            if name == "Unknown":
                # Use hysteresis for state changes
                if self.last_known_state != "unknown":
                    # Require multiple unknown detections before changing state
                    if not hasattr(self, 'unknown_count'):
                        self.unknown_count = 0
                    self.unknown_count += 1

                    if self.unknown_count >= 3:  # Require 3 consecutive unknown detections
                        self.last_known_state = "unknown"
                        self.unknown_count = 0

                        # Check if enough time has passed since last unknown alert
                        if not hasattr(self, 'last_unknown_alert') or (current_time - self.last_unknown_alert) > 5.0:
                            logger.warning(f"Unknown person detected ({confidence:.3f})")

                            # Activate Athena for unknown person alert
                            if self.athena:
                                self.athena.activate_by_cv_unknown()
                                logger.info("Activated Athena for unknown person alert")
                                self.last_unknown_alert = current_time
                            else:
                                logger.warning("Athena not available for unknown person alert")
                else:
                    # Reset unknown count when we stay in unknown state
                    self.unknown_count = 0
                break

        # Check for known residents (only if no unknown person was detected)
        for name, face_id, confidence in recognitions:
            if name != "Unknown":
                # Use hysteresis for known person identification
                confidence_threshold_high = self.confidence_threshold  # 0.6 for initial detection
                confidence_threshold_low = self.confidence_threshold - 0.2  # 0.4 to maintain identification

                should_identify = False

                if self.last_known_state == name:
                    # Already identified this person - use lower threshold to maintain state
                    should_identify = confidence >= confidence_threshold_low
                else:
                    # New person or different person - use higher threshold for initial detection
                    should_identify = confidence >= confidence_threshold_high

                if should_identify:
                    # Update state
                    self.last_known_state = name
                    self.last_known_confidence = confidence
                    if hasattr(self, 'unknown_count'):
                        self.unknown_count = 0

                    # Check if this is a new resident or enough time has passed
                    if (not hasattr(self, 'last_welcome_resident') or
                        self.last_welcome_resident != name or
                        (current_time - getattr(self, 'last_welcome_time', 0)) > 30.0):  # 30 second cooldown

                        logger.info(f"High confidence recognition: {name} ({confidence:.3f})")

                        # Activate Athena with resident detection
                        if self.athena:
                            self.athena.activate_by_cv(resident_name=name)
                            logger.info(f"Activated Athena for resident: {name}")
                            self.last_welcome_resident = name
                            self.last_welcome_time = current_time
                        else:
                            logger.warning("Athena not available for activation")
                break

    def play_welcome_feedback(self, person_name: str, confidence: float):
        """
        Play welcome message and ping sound for successful recognition.

        Args:
            person_name: Name of recognized person
            confidence: Confidence score of recognition
        """
        try:
            # Play ping sound
            self.play_ping_sound()

            # Text-to-speech welcome message
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(f"Welcome, {person_name}")

            logger.info(f"Played welcome message for {person_name}")

        except ImportError:
            # Fallback for non-Windows systems
            print(f"🗣️  Welcome, {person_name}")
        except Exception as e:
            logger.error(f"Error in voice feedback: {e}")
            print(f"🗣️  Welcome, {person_name}")

    def play_ping_sound(self):
        """Play a ping sound for successful recognition."""
        try:
            import pygame
            pygame.mixer.init()

            # Create a simple ping sound programmatically
            sample_rate = 44100
            duration = 0.3
            frequency = 800

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.3
            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 20))
            wave = wave * envelope

            wave = (wave * 32767).astype(np.int16)
            sound = pygame.mixer.Sound(wave.tobytes())
            sound.play()

        except Exception as e:
            logger.error(f"Error playing ping sound: {e}")

    def run_face_recognition(self):
        """Run the face recognition system with voice feedback."""
        try:
            print("👤 Starting face recognition system...")

            # Test camera
            ret, test_frame = self.camera.read()
            if not ret:
                print("❌ Camera test failed")
                return

            print(f"✅ Camera ready: {test_frame.shape[1]}x{test_frame.shape[0]}")
            print(f"📊 Known faces in database: {len(self.known_faces)}")
            print(f"🎯 Confidence threshold: {self.confidence_threshold}")
            print("🎵 High confidence recognition will trigger voice welcome")
            print("❌ Press 'q' to quit")

            consecutive_failures = 0
            max_consecutive_failures = 5

            while self.is_running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"❌ Camera failed {consecutive_failures} times consecutively, stopping")
                        break
                    print(f"⚠️ Camera frame capture failed (attempt {consecutive_failures}/{max_consecutive_failures})")
                    time.sleep(0.5)  # Wait before retrying
                    continue

                # Reset failure counter on successful capture
                consecutive_failures = 0

                # Flip frame for mirror view
                frame = cv2.flip(frame, 1)

                try:
                    # Get face identifications
                    identifications = self.identify_faces(frame)

                    # Check for high confidence and provide voice feedback
                    self.check_confidence_and_respond(identifications)

                    # Draw results on frame
                    output = self.draw_detections_with_identification(frame, identifications)

                    # Add system status
                    status_text = f"Voice+CV System | DB: {len(self.known_faces)} faces"
                    cv2.putText(output, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                except Exception as e:
                    print(f"❌ Error in face recognition: {e}")
                    output = frame

                # Show result
                cv2.imshow('Complete Voice-Controlled CV System', output)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
        finally:
            if self.camera:
                self.camera.release()

    def draw_detections_with_identification(self, image: np.ndarray, identifications: List[Tuple[str, str, float]]) -> np.ndarray:
        """
        Draw bounding boxes with identification results on the image.

        Args:
            image: Input image in BGR format
            identifications: List of (name, face_id, similarity_score) tuples

        Returns:
            Image with drawn detections and identification labels
        """
        output = image.copy()

        # Detect faces for drawing bounding boxes
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for i, ((x, y, w, h), (name, face_id, score)) in enumerate(zip(faces, identifications)):
            # Choose color based on identification result
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({score:.2f})"

            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label
            text_y = y - 10 if y > 20 else y + 20
            cv2.putText(output, label, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return output

    def get_builtin_camera_index(self):
        """
        Try to find a working camera index on Windows.
        Returns the index of the first working camera, or 0 if not found.
        """
        # On Windows, try the most common camera indices
        for i in range(3):  # Check first 3 indices (0, 1, 2)
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend for better Windows compatibility
                if cap.isOpened():
                    # Test multiple frames to ensure stable access
                    success_count = 0
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            success_count += 1
                        time.sleep(0.1)  # Small delay between tests

                    cap.release()

                    if success_count >= 2:  # Require at least 2 successful frames
                        print(f"Successfully opened camera at index {i} (DirectShow backend)")
                        return i
            except Exception as e:
                print(f"Error testing camera {i}: {e}")
                continue

        print("Warning: Could not find any working camera, trying index 0 with default backend")
        return 0

    def run_system(self):
        """Run the complete voice-controlled CV system."""
        print("🚀 Complete Voice-Controlled Computer Vision System")
        print("=" * 60)
        print("🎤 Voice Commands + 👤 Face Recognition + 🗣️  Audio Feedback")
        print("=" * 60)

        self.is_running = True

        try:
            # Initialize camera
            print("📷 Initializing camera...")
            camera_index = self.get_builtin_camera_index()
            self.camera = cv2.VideoCapture(camera_index)

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            # Verify the camera opened successfully
            if not self.camera.isOpened():
                print("❌ Error: Could not open the camera")
                return

            # Test frame capture
            ret, test_frame = self.camera.read()
            if not ret:
                print("❌ Error: Could not capture a test frame")
                return

            print("✅ All systems initialized successfully!")
            print(f"📊 Database loaded with {len(self.known_faces)} known faces")
            print(f"🎯 Confidence threshold: {self.confidence_threshold} ({int(self.confidence_threshold*100)}%+)")
            print(f"🎵 ≥{int(self.confidence_threshold*100)}% confidence → 'Welcome [Name]' + Athena ready for 'Athena' commands")
            print(f"🚨 <{int(self.confidence_threshold*100)}% confidence → 'Please identify yourself' + beep")
            print("🎤 In conversation: Say 'Athena [command]' or 'goodbye' to end")
            print("⏰ Welcome cooldown: 30s | Unknown alert cooldown: 5s")
            print("❌ Press 'q' to quit")

            # Run the face recognition system
            self.run_face_recognition()

        except Exception as e:
            print(f"❌ Error in system: {e}")
        finally:
            self.is_running = False
            print("👋 System shutdown complete.")

def main():
    """Main function to run the complete system."""
    try:
        # Create and run the complete system
        system = CompleteVoiceCVSystem()
        system.run_system()

    except Exception as e:
        print(f"❌ Error in main system: {e}")

if __name__ == "__main__":
    main()
