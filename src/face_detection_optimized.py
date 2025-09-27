import argparse
import cv2
import numpy as np
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFaceDetector:
    def __init__(self, min_face_size: int = 50, confidence_threshold: float = 0.7):
        """
        Initialize the optimized face detector with OpenCV's DNN module.
        Args:
            min_face_size: Minimum face size in pixels (default: 50)
            confidence_threshold: Minimum confidence threshold for detections (default: 0.7)
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.input_size = (300, 300)  # Standard input size for the model
        self.scale = 1.0
        self.mean = (104.0, 177.0, 123.0)  # Mean subtraction values for the model
        
        # Initialize the DNN model
        self.net = self._initialize_model()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_model(self):
        """Initialize the face detection model."""
        # Try different model paths
        model_paths = [
            'models/opencv_face_detector_uint8.pb',
            'opencv_face_detector_uint8.pb',
            os.path.join(cv2.data.haarcascades, '..', 'opencv_face_detector_uint8.pb')
        ]
        
        config_paths = [
            'models/opencv_face_detector.pbtxt',
            'opencv_face_detector.pbtxt',
            os.path.join(cv2.data.haarcascades, '..', 'opencv_face_detector.pbtxt')
        ]
        
        model_path = None
        config_path = None
        
        # Find model and config files
        for mp in model_paths:
            if os.path.exists(mp):
                model_path = mp
                break
                
        for cp in config_paths:
            if os.path.exists(cp):
                config_path = cp
                break
                
        if not model_path or not config_path:
            # Download the model if not found
            logger.warning("Model files not found, attempting to download...")
            try:
                import urllib.request
                os.makedirs('models', exist_ok=True)
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt',
                    'models/opencv_face_detector.pbtxt')
                urllib.request.urlretrieve(
                    'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb',
                    'models/opencv_face_detector_uint8.pb')
                model_path = 'models/opencv_face_detector_uint8.pb'
                config_path = 'models/opencv_face_detector.pbtxt'
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise FileNotFoundError("Could not load face detection model")
        
        # Load the model
        net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        
        # Try to use OpenCV's DNN backend for better performance
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Force CPU for x86 optimization
            logger.info("Using OpenCV DNN backend with CPU")
        except Exception as e:
            logger.warning(f"Could not set DNN backend: {e}")
            
        return net
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame for face detection."""
        # Resize the frame to the model's expected sizing
        resized = cv2.resize(frame, self.input_size)
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            resized, 
            scalefactor=self.scale,
            size=self.input_size,
            mean=self.mean,
            swapRB=False,  # BGR to RGB
            crop=False
        )
        return blob
    
    def _postprocess_detections(self, detections: np.ndarray, frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
        """Postprocess the raw detections from the network."""
        height, width = frame_shape[:2]
        results = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding boxes fall within the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width - 1, endX)
                endY = min(height - 1, endY)
                
                # Calculate width and height
                w = endX - startX
                h = endY - startY
                
                # Filter by minimum face size
                if w >= self.min_face_size and h >= self.min_face_size:
                    results.append((startX, startY, w, h, float(confidence)))
        
        # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
        if results:
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h, _) in results])
            confidences = np.array([c for (_, _, _, _, c) in results])
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                confidences.tolist(),
                self.confidence_threshold,
                0.4  # NMS threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                results = [results[i] for i in indices]
        
        return results
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in the input frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            List of detected faces as (x, y, width, height, confidence) tuples
        """
        try:
            # Preprocess the frame
            blob = self._preprocess_frame(frame)
            
            # Perform the forward pass
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Post-process the detections
            results = self._postprocess_detections(detections, frame.shape)
            
            logger.debug(f"Detected {len(results)} faces")
            return results
            
        except Exception as e:
            logger.error(f"Error in detect_faces: {str(e)}", exc_info=True)
            return []
    
    def detect_faces_async(self, frame: np.ndarray):
        """
        Asynchronously detect faces in the input frame.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Future object that will contain the detection results
        """
        return self.executor.submit(self.detect_faces, frame.copy())
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


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
        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"{score:.2f}"
        text_y = y - 10 if y > 20 else y + 20
        cv2.putText(output, label, (x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return output


def get_builtin_camera_index():
    """Try to find the built-in camera index."""
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
    parser = argparse.ArgumentParser(description='Optimized Face Detection using OpenCV DNN')
    parser.add_argument('--min-face-size', type=int, default=50,
                      help='Minimum face size in pixels (default: 50)')
    parser.add_argument('--confidence', type=float, default=0.7,
                      help='Confidence threshold (default: 0.7)')
    parser.add_argument('--frame-skip', type=int, default=1,
                      help='Number of frames to skip between detections (default: 1)')
    args = parser.parse_args()
    
    # Initialize face detector
    try:
        detector = OptimizedFaceDetector(
            min_face_size=args.min_face_size,
            confidence_threshold=args.confidence
        )
        logger.info("Optimized face detector initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing face detector: {e}", exc_info=True)
        return
    
    try:
        # Initialize camera
        print("Initializing camera...")
        camera_index = get_builtin_camera_index()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open the camera.")
            return
        
        # Test frame capture
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not capture a test frame from the camera.")
            cap.release()
            return
        
        print(f"Camera initialized. Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
        print("Press 'q' to quit the application.")
        
        frame_count = 0
        fps = 0
        last_time = time.time()
        
        # For async processing
        future = None
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip the frame for a more intuitive selfie view
            frame = cv2.flip(frame, 1)
            
            # Only process every args.frame_skip-th frame
            if frame_count % (args.frame_skip + 1) == 0:
                # If we have a previous async detection, get the results
                if future is not None:
                    try:
                        detections = future.result(timeout=0.1)
                    except Exception as e:
                        logger.warning(f"Error getting detection results: {e}")
                        detections = []
                
                # Start a new async detection
                future = detector.detect_faces_async(frame)
            
            # Draw the most recent detections
            output = draw_detections(frame, detections)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - last_time)
                last_time = current_time
            
            cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow('Optimized Face Detection', output)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        detector.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
