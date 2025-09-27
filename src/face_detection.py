import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str = None, threshold: float = 0.5, min_face_size: int = 50):
        """
        Initialize the face detector with a TensorFlow Lite model.
        
        Args:
            model_path: Path to the TFLite model file. If None, uses the default model.
            threshold: Confidence threshold for face detection (0-1, default: 0.5).
            min_face_size: Minimum face size in pixels (default: 50).
        """
        self.threshold = threshold
        self.min_face_size = min_face_size  # Store as single integer value
        
        # Set default model path
        default_model = "/Users/ankithnagabandi/Documents/Qualcomm Hackathon/AthenaSecurityGuardian/src/models/face_det_lite-lightweight-face-detection-float.tflite"
        model_to_use = model_path if model_path else default_model
        
        try:
            logger.info(f"Loading model from: {model_to_use}")
            # Initialize TFLite interpreter
            self.interpreter = tf.lite.Interpreter(model_path=model_to_use)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Log model details
            logger.info(f"Input details: {self.input_details}")
            logger.info(f"Output details: {self.output_details}")
            
            # Get input shape details
            self.input_shape = self.input_details[0]['shape']
            self.height = self.input_shape[1]
            self.width = self.input_shape[2]
            self.channels = self.input_shape[3]
            
            logger.info(f"Model input shape: {self.input_shape} (HxWxC: {self.height}x{self.width}x{self.channels})")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_default_model(self):
        """Load a default face detection model if none is provided."""
        # This is a placeholder for a default model
        # In a real implementation, you would provide a pre-trained model
        raise FileNotFoundError("No model path provided and no default model available.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for the model.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            Preprocessed image in the format expected by the model (grayscale, float32, normalized)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:  # BGR
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = image
            
            # Resize to model's expected sizing (480x640 as per model input shape)
            resized = cv2.resize(gray, (self.width, self.height))
            
            # Convert to float32 and normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Add channel and batch dimensions (shape: [1, height, width, 1])
            return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
            
        except Exception as e:
            logger.error(f"Error in preprocess_image: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            List of detected faces as (x, y, width, height, confidence) tuples
        """
        try:
            # Preprocess the image
            input_data = self.preprocess_image(image)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensors
            heatmap = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Heatmap
            boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]    # Bounding boxes
            
            # Debug: Print output shapes and sample values
            logger.debug(f"Heatmap shape: {heatmap.shape}, max: {np.max(heatmap):.2f}, min: {np.min(heatmap):.2f}")
            logger.debug(f"Boxes shape: {boxes.shape}")
            
            # Process detections
            detections = []
            height, width = image.shape[:2]
            
            # Find all scores above threshold
            scores_above_threshold = heatmap[..., 0] > self.threshold
            y_indices, x_indices = np.where(scores_above_threshold)
            
            for y, x in zip(y_indices, x_indices):
                score = heatmap[y, x, 0]
                
                # Get bounding box coordinates (normalized [0,1])
                # The model might output [y_center, x_center, height, width] or [y1, x1, y2, x2]
                # Let's try both formats
                if boxes[y, x, 2] <= 1.0 and boxes[y, x, 3] <= 1.0:  # Likely [y_center, x_center, h, w]
                    y_center, x_center, h, w = boxes[y, x]
                    x1 = max(0, int((x_center - w/2) * width))
                    y1 = max(0, int((y_center - h/2) * height))
                    x2 = min(width - 1, int((x_center + w/2) * width))
                    y2 = min(height - 1, int((y_center + h/2) * height))
                else:  # Likely [y1, x1, y2, x2]
                    y1, x1, y2, x2 = boxes[y, x]
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)
                
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                
                # Only add if the face is large enough
                if w >= self.min_face_size and h >= self.min_face_size:
                    detections.append((x1, y1, w, h, float(score)))
                    logger.debug(f"Added detection: ({x1}, {y1}, {w}, {h}) with score {score:.2f}")
                else:
                    logger.debug(f"Skipped small detection: ({x1}, {y1}, {w}, {h}) is smaller than min size {self.min_face_size}")
            
            logger.debug(f"Found {len(detections)} faces")
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
    Try to find the built-in camera index by checking camera names.
    Returns the index of the built-in camera, or 0 if not found.
    """
    try:
        # List all available cameras
        index = 0
        available_cameras = []
        
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                break
                
            # Get camera name (works on macOS with AVFoundation)
            camera_name = f"Camera {index}"
            try:
                # Try to get the camera name (might not work on all platforms)
                if hasattr(cv2, 'CAP_PROP_POS_MSEC'):
                    camera_name = cap.getBackendName()
            except:
                pass
                
            available_cameras.append((index, camera_name))
            cap.release()
            index += 1
        
        # Try to find built-in camera (usually contains 'FaceTime' or 'Built-in' in the name on macOS)
        for idx, name in available_cameras:
            if any(term in name for term in ['FaceTime', 'Built-in', 'Face', 'iSight']):
                print(f"Found built-in camera: {name} (index {idx})")
                return idx
        
        # If no built-in camera found, return the first available one
        if available_cameras:
            print(f"Using first available camera: {available_cameras[0][1]} (index {available_cameras[0][0]})")
            return available_cameras[0][0]
            
    except Exception as e:
        print(f"Error detecting cameras: {e}")
    
    # Fallback to index 0 if anything fails
    print("Could not detect cameras, defaulting to index 0")
    return 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Detection using TensorFlow Lite')
    parser.add_argument('--model', type=str, 
                      default='/Users/ankithnagabandi/Documents/Qualcomm Hackathon/AthenaSecurityGuardian/src/models/face_det_lite-lightweight-face-detection-float.tflite',
                      help='Path to TFLite model')
    parser.add_argument('--threshold', type=float, default=0.3, 
                      help='Detection confidence threshold (0-1, default: 0.3)')
    parser.add_argument('--min-face-size', type=int, default=50,
                      help='Minimum face size in pixels (default: 50)')
    args = parser.parse_args()
    
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize face detector
    try:
        # Convert min_face_size to integer
        min_face_size = int(args.min_face_size)
        print(f"Initializing face detector with threshold={args.threshold}, min_face_size={min_face_size}")
        print(f"Using model: {args.model}")
        
        detector = FaceDetector(
            model_path=args.model, 
            threshold=float(args.threshold),
            min_face_size=min_face_size
        )
        
        # Print model input/output details
        print("\nModel Input Details:")
        for i, detail in enumerate(detector.input_details):
            print(f"  Input {i}: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")
            
        print("\nModel Output Details:")
        for i, detail in enumerate(detector.output_details):
            print(f"  Output {i}: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")
            
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        print("Please check if the model file exists and is a valid TFLite model.")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize camera
    print("Initializing camera...")
    
    # Get built-in camera index
    camera_index = get_builtin_camera_index()
    
    # Initialize camera with AVFoundation (works best on macOS)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
