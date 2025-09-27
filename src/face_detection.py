import argparse
import cv2
import numpy as np
import os
import time
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use OpenCV's built-in Haar Cascade for face detection
class FaceDetector:
    def __init__(self, min_face_size: int = 50):
        """
        Initialize the face detector with OpenCV's Haar Cascade.
        Args:
            min_face_size: Minimum face size in pixels (default: 50).
        """
        self.min_face_size = min_face_size

        # Try to find the Haar Cascade file
        cascade_paths = [
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ]

        self.face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if self.face_cascade.empty():
                    logger.warning(f"Failed to load cascade file: {path}")
                    continue
                logger.info(f"Loaded Haar Cascade from: {path}")
                break

        if self.face_cascade is None:
            error_msg = "Could not load Haar Cascade file. Make sure OpenCV is properly installed."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the input image using Haar Cascade.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            List of detected faces as (x, y, width, height, confidence) tuples
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Convert to expected format (add confidence score of 1.0 since Haar doesn't provide it)
            detections = [(x, y, w, h, 1.0) for (x, y, w, h) in faces]

            logger.debug(f"Detected {len(detections)} faces")
            return detections

        except Exception as e:
            logger.error(f"Error in detect_faces: {str(e)}", exc_info=True)
            return []

def draw_detections(image: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
    """
    Draw bounding boxes and confidence scores on the image.

    Args:
        image: Input image in BGR format
        detections: List of detections in (x, y, w, h, score) format

    Returns:
        Image with drawn detections
    """
    output = image.copy()

    for (x, y, w, h, score) in detections:
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1] - 1, x + w), min(image.shape[0] - 1, y + h)

        # Only draw if the box is valid
        if x1 < x2 and y1 < y2:
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with confidence
            label = f"{score:.2f}"
            # Ensure text is within image bounds
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(output, label, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output

def get_builtin_camera_index():
    """
    Try to find the built-in camera index on macOS.
    Returns the index of the built-in camera, or 0 if not found.
    """
    # On macOS, the built-in camera is usually at index 0
    for i in range(3):  # Check first 3 indices
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
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
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV Haar Cascade')
    parser.add_argument('--min-face-size', type=int, default=50,
                      help='Minimum face size in pixels (default: 50)')
    args = parser.parse_args()

    # Initialize face detector
    try:
        min_face_size = int(args.min_face_size)
        logger.info(f"Initializing face detector with min_face_size={min_face_size}")

        detector = FaceDetector(min_face_size=min_face_size)
        logger.info("Face detector initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing face detector: {e}", exc_info=True)
        return

    # Initialize camera - force using AVFoundation and the built-in camera
    print("Initializing built-in camera...")

    # Try to get the built-in camera index
    camera_index = get_builtin_camera_index()

    # Force AVFoundation backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the built-in camera.")
        print("Please check the following:")
        print("1. No other application is using the camera")
        print("2. Camera permissions are granted in System Preferences > Security & Privacy > Camera")
        return

    # Test frame capture
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not capture a test frame from the camera.")
        cap.release()
        return

    print(f"Camera initialized successfully. Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a more intuitive selfie view
        frame = cv2.flip(frame, 1)

        # Detect faces
        detections = detector.detect_faces(frame)

        # Draw detections
        output = draw_detections(frame, detections)

        # Show result
        cv2.imshow('Face Detection', output)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
