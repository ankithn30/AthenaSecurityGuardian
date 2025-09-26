import cv2
import numpy as np
import onnxruntime as ort
import time
from ultralytics import YOLO
import psutil
import os

def load_yolo_onnx(model_path='yolov8n.onnx'):
    """Load YOLOv8 ONNX model"""
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except:
        print(f"ONNX model not found at {model_path}. Exporting from PyTorch model...")
        # Export YOLOv8n to ONNX
        model = YOLO('yolov8n.pt')
        model.export(format='onnx')
        session = ort.InferenceSession('yolov8n.onnx', providers=['CPUExecutionProvider'])
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

def benchmark_inference(session, num_frames=100):
    """Benchmark YOLO inference performance"""
    # Create dummy input
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # Warm up
    for _ in range(10):
        session.run(None, {session.get_inputs()[0].name: dummy_input})

    # Benchmark
    start_time = time.time()
    for _ in range(num_frames):
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_frames
    fps = 1.0 / avg_inference_time

    return fps, avg_inference_time

def benchmark_webcam(session, duration=30):
    """Benchmark webcam performance for specified duration"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None

    frame_times = []
    frame_count = 0
    start_time = time.time()

    print(f"Benchmarking webcam for {duration} seconds...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Preprocess
        input_tensor = preprocess_frame(frame)

        # Inference
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})

        frame_end = time.time()
        inference_time = frame_end - frame_start
        frame_times.append(inference_time)

        frame_count += 1

        # Display progress
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(".1f")

    cap.release()

    if frame_times:
        avg_inference_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_inference_time
        min_fps = 1.0 / np.max(frame_times)
        max_fps = 1.0 / np.min(frame_times)

        return {
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'avg_inference_time': avg_inference_time,
            'total_frames': frame_count
        }
    else:
        return None

def get_system_info():
    """Get basic system information"""
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'platform': os.uname().sysname if hasattr(os, 'uname') else 'Unknown'
    }

def main():
    print("YOLOv8 Webcam Performance Benchmark")
    print("=" * 40)

    # System info
    sys_info = get_system_info()
    print(f"Platform: {sys_info['platform']}")
    print(f"CPU Cores: {sys_info['cpu_count']}")
    print(f"CPU Usage: {sys_info['cpu_percent']}%")
    print(f"Memory Usage: {sys_info['memory_percent']}%")
    print()

    # Load model
    print("Loading YOLOv8n ONNX model...")
    session = load_yolo_onnx()
    print("Model loaded successfully.")
    print()

    # Dummy benchmark
    print("Running dummy inference benchmark...")
    fps_dummy, avg_time_dummy = benchmark_inference(session)
    print(".2f")
    print(".4f")
    print()

    # Webcam benchmark
    print("Running webcam benchmark (30 seconds)...")
    webcam_results = benchmark_webcam(session, duration=30)

    if webcam_results:
        print("Webcam Benchmark Results:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".4f")
        print(f"Total Frames Processed: {webcam_results['total_frames']}")
        print()

        if webcam_results['avg_fps'] > 30:
            print("✅ SUCCESS: Achieved >30 FPS target!")
        else:
            print("❌ FAILED: Did not reach 30 FPS target.")
            print(".2f")
    else:
        print("❌ Webcam benchmark failed.")

    print("\nBenchmark complete.")

if __name__ == '__main__':
    main()
