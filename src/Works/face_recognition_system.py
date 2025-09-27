import argparse
import cv2
import numpy as np
import os
import time
import logging
import pickle
import face_recognition
from typing import Tuple, List, Optional, Dict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir: str = "known_faces", tolerance: float = 0.6):
        """
        Initialize the face recognition system.

        Args:
            known_faces_dir: Directory containing known faces database
            tolerance: How much distance between faces to consider it a match (lower = stricter)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = {}

        # Create known faces directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)

        # Load known faces database
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database directory."""
        if not os.path.exists(self.known_faces_dir):
            logger.info(f"Creating known faces directory: {self.known_faces_dir}")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.known_faces_dir, filename)

                # Extract name from filename (remove extension)
                name = os.path.splitext(filename)[0]

                try:
                    # Load image and get face encoding
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        logger.info(f"Loaded known face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")

                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Loaded {len(self.known_face_encodings)} known faces")

    def add_known_face(self, image_path: str, name: str) -> bool:
        """
        Add a new face to the known faces database.

        Args:
            image_path: Path to the image containing the face
            name: Name identifier for the person

        Returns:
            True if successfully added, False otherwise
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                logger.error(f"No face found in {image_path}")
                return False

            # Save the image to known_faces directory
            filename = f"{name}.jpg"
            save_path = os.path.join(self.known_faces_dir, filename)

            # Save a copy of the image
            cv2_image = cv2.imread(image_path)
            cv2.imwrite(save_path, cv2_image)

            # Add to memory
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)

            logger.info(f"Added known face: {name}")
            return True

        except Exception as e:
            logger.error(f"Error adding face {name}: {e}")
            return False

    def recognize_faces(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Recognize faces from their encodings.

        Args:
            face_encodings: List of face encodings to identify

        Returns:
            List of (name, distance) tuples for each face
        """
        recognized_faces = []

        for face_encoding in face_encodings:
            # Compare with known faces
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )

                # Find the best match
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                if best_distance <= self.tolerance:
                    name = self.known_face_names[best_match_index]
                    recognized_faces.append((name, best_distance))
                else:
                    recognized_faces.append(("Unknown", best_distance))
            else:
                recognized_faces.append(("No known faces in database", 0.0))

        return recognized_faces

def draw_detections_with_recognition(
    image: np.ndarray,
    detections: List[Tuple[int, int, int, int, float]],
    recognitions: List[Tuple[str, float]]
) -> np.ndarray:
    """
    Draw bounding boxes with recognition results on the image.

    Args:
        image: Input image in BGR format
        detections: List of detections in (x, y, w, h, score) format
        recognitions: List of (name, distance) tuples

    Returns:
        Image with drawn detections and recognition labels
    """
    output = image.copy()

    for i, ((x, y, w, h, score), (name, distance)) in enumerate(zip(detections, recognitions)):
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1] - 1, x + w), min(image.shape[0] - 1, y + h)

        # Only draw if the box is valid
        if x1 < x2 and y1 < y2:
            # Choose color based on recognition result
            if name == "Unknown" or "No known faces" in name:
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({distance:.2f})"

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(output, label, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output

def get_builtin_camera_index():
    """
    Try to find a working camera index on Windows.
    Returns the index of the first working camera, or 0 if not found.
    """
    # On Windows, try the most common camera indices
    for i in range(3):  # Check first 3 indices (0, 1, 2)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera at index {i}")
                cap.release()
                return i
            cap.release()

    print("Warning: Could not find any working camera, trying index 0")
    return 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--known-faces-dir', type=str, default='known_faces',
                      help='Directory containing known faces database')
    parser.add_argument('--tolerance', type=float, default=0.6,
                      help='Recognition tolerance (lower = stricter)')
    parser.add_argument('--add-face', type=str, nargs=2, metavar=('IMAGE_PATH', 'NAME'),
                      help='Add a new face to the database')
    args = parser.parse_args()

    # Initialize face recognition system
    try:
        recognition_system = FaceRecognitionSystem(
            known_faces_dir=args.known_faces_dir,
            tolerance=args.tolerance
        )
        logger.info("Face recognition system initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing face recognition system: {e}", exc_info=True)
        return

    # Handle adding a new face if requested
    if args.add_face:
        image_path, name = args.add_face
        if os.path.exists(image_path):
            success = recognition_system.add_known_face(image_path, name)
            if success:
                print(f"Successfully added {name} to the database!")
            else:
                print(f"Failed to add {name} to the database.")
        else:
            print(f"Image file not found: {image_path}")
        return

    # Initialize camera
    print("Initializing camera...")
    camera_index = get_builtin_camera_index()
    cap = cv2.VideoCapture(camera_index)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        print("Please check the following:")
        print("1. No other application is using the camera")
        print("2. Camera permissions are granted in Windows Settings > Privacy & security > Camera")
        print("3. The camera is not physically blocked or disconnected")
        print("4. Try running the application as administrator")
        return

    # Test frame capture
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not capture a test frame from the camera.")
        cap.release()
        return

    print(f"Camera initialized successfully. Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("Press 'q' to quit, 'a' to add current face to database")
    print(f"Known faces in database: {len(recognition_system.known_face_encodings)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a more intuitive selfie view
        frame = cv2.flip(frame, 1)

        try:
            # Find all faces in the current frame and get their encodings
            face_encodings = face_recognition.face_encodings(frame)

            if face_encodings:
                # Recognize faces
                recognitions = recognition_system.recognize_faces(face_encodings)

                # Create dummy detections for drawing (we'll use the actual face locations)
                # For simplicity, we'll create bounding boxes from the face encodings
                detections = []
                for face_encoding in face_encodings:
                    # Get face location (this is a simplified approach)
                    # In a real implementation, you'd get the actual bounding box from face_recognition
                    top, right, bottom, left = face_recognition.face_locations(frame)[0]
                    h = bottom - top
                    w = right - left
                    detections.append((left, top, w, h, 1.0))

                # Draw detections with recognition results
                output = draw_detections_with_recognition(frame, detections, recognitions)
            else:
                output = frame

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            output = frame

        # Show result
        cv2.imshow('Face Recognition', output)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') and face_encodings:
            # Add current face to database
            print("Enter name for this person:")
            name = input().strip()
            if name:
                # Save current frame as the known face
                temp_image = f"temp_face_{int(time.time())}.jpg"
                cv2.imwrite(temp_image, frame)

                success = recognition_system.add_known_face(temp_image, name)
                if success:
                    print(f"Added {name} to database!")
                else:
                    print("Failed to add face to database.")

                # Clean up temp file
                if os.path.exists(temp_image):
                    os.remove(temp_image)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
