import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import pyttsx3
from pathlib import Path
from ultralytics import YOLO

import ssl
import urllib.request

# Constants for face detection
MODELS_DIR = '../models'
FACE_CONFIDENCE_THRESHOLD = 0.7

# Download face detection model if not exists
def download_face_model():
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Define paths
    prototxt_path = os.path.join(MODELS_DIR, 'deploy.prototxt')
    model_path = os.path.join(MODELS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    # Check if files already exist
    if os.path.exists(prototxt_path) and os.path.exists(model_path):
        return prototxt_path, model_path
        
    print("Downloading face detection models...")
    
    # Create unverified SSL context
    ssl_context = ssl._create_unverified_context()
    
    try:
        # Download prototxt
        if not os.path.exists(prototxt_path):
            print("Downloading prototxt...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
                prototxt_path,
                context=ssl_context
            )
            
        # Download model
        if not os.path.exists(model_path):
            print("Downloading model... (this may take a while)")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                model_path,
                context=ssl_context
            )
            
    except Exception as e:
        print(f"Error downloading models: {e}")
        # Clean up partially downloaded files
        if os.path.exists(prototxt_path) and not os.path.exists(model_path):
            try:
                os.remove(prototxt_path)
            except:
                pass
        raise
    
    return prototxt_path, model_path

def load_face_detector():
    """Load the face detection model"""
    try:
        prototxt_path, model_path = download_face_model()
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net
    except Exception as e:
        print(f"Warning: Could not load face detection model: {e}")
        print("Running in YOLO-only mode (face detection disabled)")
        return None

def load_yolo_onnx(model_path='models/yolov8_det_quantized.onnx'):
    """Load YOLOv8 ONNX model"""
    try:
        session = ort.InferenceSession(model_path)
        return session
    except:
        print(f"ONNX model not found at {model_path}. Exporting from PyTorch model...")
        # Export YOLOv8n to ONNX
        model = YOLO('models/yolov8n.pt')
        model.export(format='onnx')
        session = ort.InferenceSession('../models/yolov8n.onnx')
        return session

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocess frame for YOLO inference"""
    # Resize
    img = cv2.resize(frame, input_size)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    # Transpose to CHW
    img = np.transpose(img, (2, 0, 1))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_output(output, conf_threshold=0.25, iou_threshold=0.45, input_size=(640, 640), original_size=(640, 480)):
    """Postprocess YOLOv8 output to get bounding boxes with NMS"""
    try:
        # For YOLOv8, output is a list with a single array of shape [1, num_detections, 84]
        # where 84 = [x, y, w, h, obj_conf, class_probs...]
        predictions = output[0][0]  # Remove batch dimension
        
        # Get boxes (x, y, w, h) and scores
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # obj_conf and class_probs
        
        # Get class with maximum probability for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter out low confidence detections
        mask = confidences > conf_threshold
        if not np.any(mask):
            return []
            
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert from center (x, y, w, h) to (x1, y1, x2, y2)
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * (original_size[0] / input_size[0])
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * (original_size[1] / input_size[1])
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * (original_size[0] / input_size[0])
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * (original_size[1] / input_size[1])
        
        # Convert to integers and clip to image boundaries
        x1 = np.clip(x1, 0, original_size[0] - 1).astype(int)
        y1 = np.clip(y1, 0, original_size[1] - 1).astype(int)
        x2 = np.clip(x2, 0, original_size[0] - 1).astype(int)
        y2 = np.clip(y2, 0, original_size[1] - 1).astype(int)
        
        # Apply NMS
        boxes_xyxy = np.column_stack([x1, y1, x2, y2])
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), 
            confidences.tolist(), 
            conf_threshold, 
            iou_threshold
        )
        
        # Prepare output
        filtered_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1_i, y1_i, x2_i, y2_i = boxes_xyxy[i]
                conf = float(confidences[i])
                class_id = int(class_ids[i])
                filtered_boxes.append([x1_i, y1_i, x2_i, y2_i, conf, class_id])
        
        return filtered_boxes
        
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        import traceback
        traceback.print_exc()
        return []

def speak(text):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Slower speech rate for better clarity
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        print(f"Tried to say: {text}")

def detect_faces(net, frame, min_confidence=0.5):
    """Detect faces in the frame using OpenCV's DNN module"""
    if net is None:
        return []
        
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()
        faces = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                faces.append((startX, startY, endX, endY, confidence))
        
        return faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def draw_boxes(frame, yolo_boxes, faces):
    """Draw bounding boxes on frame for both YOLO detections and faces"""
    person_detected = False
    face_detected = False
    
    # Draw YOLO detections
    for box in yolo_boxes:
        x1, y1, x2, y2, conf, class_id = box
        
        # Check if the detected object is a person (class_id 0 in YOLO)
        if int(class_id) == 0:  # 0 is typically the class ID for 'person' in YOLO
            color = (0, 255, 0)  # Green for person
            label = f"Person: {conf:.2f}"
            person_detected = True
        else:
            color = (0, 0, 255)  # Red for other objects
            label = f"Object {int(class_id)}: {conf:.2f}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw face detections
    for (x1, y1, x2, y2, confidence) in faces:
        # Draw a rectangle around the face
        color = (255, 0, 0)  # Blue for face
        label = f"Face: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        y = y1 - 15 if y1 - 15 > 15 else y1 + 15
        cv2.putText(frame, label, (x1, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        face_detected = True
    
    # Announce detections
    if person_detected and face_detected:
        speak("Person with face detected")
    elif person_detected:
        speak("Person detected")
    elif face_detected:
        speak("Face detected")
        
    return frame, person_detected or face_detected

def main():
    # Load YOLO model
    session = load_yolo_onnx()
    
    # Try to load face detector
    face_net = None
    try:
        face_net = load_face_detector()
        if face_net is not None:
            print("Face detection model loaded successfully")
    except Exception as e:
        print(f"Warning: Face detection disabled: {e}")
    
    # Variables for announcement cooldown
    last_announcement = 0
    ANNOUNCEMENT_COOLDOWN = 3  # seconds between announcements

    # List available cameras
    def list_cameras():
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

    # Try to open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    # Check if the default camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open default camera (index 0)")
        available_cameras = list_cameras()
        if available_cameras:
            print(f"Available camera indices: {available_cameras}")
            print("Trying to use the first available camera...")
            cap = cv2.VideoCapture(available_cameras[0])
            if not cap.isOpened():
                print("Error: Could not open any camera")
                return
            print(f"Successfully opened camera at index {available_cameras[0]}")
        else:
            print("No cameras found. Please connect a camera and try again.")
            return

    print("Starting YOLOv8 webcam inference. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_size = (frame.shape[1], frame.shape[0])  # width, height

        try:
            # Preprocess
            input_tensor = preprocess_frame(frame)

            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            
            # Debug: Print output shapes
            if frame_count == 0:  # Only print once
                print("Model output shapes:", [out.shape if hasattr(out, 'shape') else str(out) for out in outputs])

            # Postprocess YOLO detections
            boxes = postprocess_output(outputs, original_size=original_size)
            
            # Detect faces if face detection is available
            faces = []
            if face_net is not None:
                faces = detect_faces(face_net, frame, min_confidence=FACE_CONFIDENCE_THRESHOLD)
            
            # Draw all detections
            frame_with_boxes, detection_made = draw_boxes(frame.copy(), boxes, faces)
            
        except Exception as e:
            print(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()
            frame_with_boxes = frame.copy()
        
        # Only announce if something was detected and cooldown has passed
        current_time = time.time()
        if detection_made and (current_time - last_announcement) > ANNOUNCEMENT_COOLDOWN:
            last_announcement = current_time

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display FPS
        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('YOLOv8 Webcam Inference', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(".2f")

if __name__ == '__main__':
    main()
