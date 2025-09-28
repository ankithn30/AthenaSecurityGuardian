import argparse
import cv2
import numpy as np
import os
import time
import logging
import pickle
import json
from typing import Tuple, List, Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDatabaseSystem:
    def __init__(self, known_faces_dir: str = "known_faces"):
        """
        Initialize the face database system for manual face identification.

        Args:
            known_faces_dir: Directory containing known faces database
        """
        self.known_faces_dir = known_faces_dir
        self.known_faces = {}
        self.face_cascade = None

        # Try to load Haar Cascade
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

        # Create known faces directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)

        # Load known faces database
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the database directory."""
        if not os.path.exists(self.known_faces_dir):
            logger.info(f"Creating known faces directory: {self.known_faces_dir}")
            return

        # Load the known faces metadata
        metadata_file = os.path.join(self.known_faces_dir, "faces_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.known_faces = json.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces from database")
            except Exception as e:
                logger.error(f"Error loading faces metadata: {e}")

    def save_known_faces(self):
        """Save known faces metadata to file."""
        metadata_file = os.path.join(self.known_faces_dir, "faces_metadata.json")
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.known_faces, f, indent=2)
            logger.info("Saved known faces metadata")
        except Exception as e:
            logger.error(f"Error saving faces metadata: {e}")

    def add_face_from_image(self, image_path: str, name: str, face_id: str = None) -> bool:
        """
        Add a face from an image file to the database.

        Args:
            image_path: Path to the image containing the face
            name: Name identifier for the person
            face_id: Unique identifier for this face (auto-generated if not provided)

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                logger.error(f"No face found in {image_path}")
                return False

            if len(faces) > 1:
                logger.warning(f"Multiple faces found in {image_path}, using the first one")

            # Use the first detected face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            # Generate face ID if not provided
            if face_id is None:
                face_id = f"{name}_{len(self.known_faces)}"

            # Save face image
            face_filename = f"{face_id}.jpg"
            face_path = os.path.join(self.known_faces_dir, face_filename)
            cv2.imwrite(face_path, face_roi)

            # Store metadata
            self.known_faces[face_id] = {
                'name': name,
                'image_path': face_path,
                'date_added': time.time(),
                'face_size': (int(w), int(h)),
                'face_position': (int(x), int(y))
            }

            # Save updated database
            self.save_known_faces()

            logger.info(f"Added face: {name} (ID: {face_id})")
            return True

        except Exception as e:
            logger.error(f"Error adding face {name}: {e}")
            return False

    def identify_faces(self, image: np.ndarray) -> List[Tuple[str, str, float, Tuple[int, int, int, int]]]:
        """
        Identify faces in the input image by comparing with known faces.

        Args:
            image: Input image in BGR format

        Returns:
            List of (name, face_id, similarity_score, (x, y, w, h)) tuples for each detected face
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

                # Simple similarity comparison with known faces
                best_match = None
                best_score = 0.0
                best_face_id = None

                for face_id, metadata in self.known_faces.items():
                    known_face_path = metadata['image_path']

                    if os.path.exists(known_face_path):
                        known_face = cv2.imread(known_face_path, cv2.IMREAD_GRAYSCALE)

                        if known_face is not None and known_face.shape[0] > 0 and known_face.shape[1] > 0:
                            # Resize faces to same size for comparison
                            try:
                                resized_current = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))

                                # Simple similarity using histogram comparison (more reliable)
                                hist_current = cv2.calcHist([resized_current], [0], None, [256], [0, 256])
                                hist_known = cv2.calcHist([known_face], [0], None, [256], [0, 256])
                                similarity = cv2.compareHist(hist_current, hist_known, cv2.HISTCMP_CORREL)

                                if similarity > best_score and similarity > 0.4:  # Reasonable threshold
                                    best_score = similarity
                                    best_match = metadata['name']
                                    best_face_id = face_id

                            except Exception as e:
                                logger.debug(f"Error comparing faces: {e}")
                                continue

                if best_match:
                    identifications.append((best_match, best_face_id, best_score, (x, y, w, h)))
                else:
                    identifications.append(("Unknown", "unknown", 0.0, (x, y, w, h)))

            return identifications

        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return []

def draw_detections_with_identification(
    image: np.ndarray,
    identifications: List[Tuple[str, str, float]]
) -> np.ndarray:
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
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
    parser = argparse.ArgumentParser(description='Face Database System')
    parser.add_argument('--known-faces-dir', type=str, default='known_faces',
                      help='Directory containing known faces database')
    parser.add_argument('--add-face', type=str, nargs=2, metavar=('IMAGE_PATH', 'NAME'),
                      help='Add a face image to the database')
    args = parser.parse_args()

    # Initialize face database system
    try:
        database_system = FaceDatabaseSystem(known_faces_dir=args.known_faces_dir)
        logger.info("Face database system initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing face database system: {e}", exc_info=True)
        return

    # Handle adding a face if requested
    if args.add_face:
        image_path, name = args.add_face
        if os.path.exists(image_path):
            success = database_system.add_face_from_image(image_path, name)
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
    print("Press 'q' to quit")
    print(f"Known faces in database: {len(database_system.known_faces)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a more intuitive selfie view
        frame = cv2.flip(frame, 1)

        try:
            # Identify faces
            identifications = database_system.identify_faces(frame)

            # Draw detections with identification results
            output = draw_detections_with_identification(frame, identifications)

        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            output = frame

        # Show result
        cv2.imshow('Face Database System', output)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
