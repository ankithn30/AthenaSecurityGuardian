import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

def load_models():
    # Load insightface model for face detection and recognition (includes MobileFaceNet backbone)
    app = FaceAnalysis(name='buffalo_l')  # buffalo_l includes detection and recognition
    app.prepare(ctx_id=0, det_size=(640, 640))  # Use CPU, adjust for GPU if available
    return app

def process_image(image_path, app):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, []

    # Detect faces and get embeddings
    faces = app.get(img)
    return img, faces

def detect_anomalies(faces, eps=0.5, min_samples=1):
    if len(faces) < 2:
        return []  # Not enough faces for clustering

    # Extract embeddings
    embeddings = np.array([face.embedding for face in faces])

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Use DBSCAN for clustering (anomaly detection)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings_scaled)
    labels = db.labels_

    # Anomalies are points labeled as -1 (noise)
    anomalies = [i for i, label in enumerate(labels) if label == -1]
    return anomalies

def validate_inference(image_path, app):
    img, faces = process_image(image_path, app)
    if img is None:
        return False

    print(f"Detected {len(faces)} faces in {image_path}")

    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        print(f"Face {i+1}: bbox {bbox}, confidence {face.det_score:.2f}")

    # Basic anomaly detection
    anomalies = detect_anomalies(faces)
    if anomalies:
        print(f"Anomalous faces detected at indices: {anomalies}")
    else:
        print("No anomalies detected")

    # Draw bounding boxes
    for face in faces:
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if anomalies and faces.index(face) in anomalies else (255, 0, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Save result
    output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

    return len(faces) > 0  # Validation: at least one face detected

def main():
    # Load models
    app = load_models()

    # Example usage: process a sample image (replace with actual path)
    # For demo, assume an image exists or download one
    image_path = 'data/sample_face.jpg'  # Replace with actual image path

    if not os.path.exists(image_path):
        print(f"Sample image {image_path} not found. Please provide an image path.")
        # For demo, you can download a sample image
        # import urllib.request
        # urllib.request.urlretrieve('https://example.com/sample_face.jpg', image_path)
        return

    # End-to-end inference validation
    success = validate_inference(image_path, app)
    if success:
        print("Inference validation successful")
    else:
        print("Inference validation failed")

if __name__ == '__main__':
    main()
