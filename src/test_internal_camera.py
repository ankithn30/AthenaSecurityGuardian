import cv2

print("Testing internal camera (index 0)...")

cap = cv2.VideoCapture(0)  # Internal camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Internal camera (index 0) not available.")
    print("1. Check System Preferences > Security & Privacy > Privacy > Camera > Allow Terminal/PyCharm.")
    print("2. Close other apps using camera (Zoom, FaceTime, etc.).")
    print("3. Restart Mac or log out/in.")
    exit(1)

print("Internal camera opened successfully!")
ret, frame = cap.read()
if ret:
    print(f"Frame captured: {frame.shape} (success!)")
    cv2.imwrite('data/internal_test.jpg', frame)
    print("Saved test image to data/internal_test.jpg")
else:
    print("Failed to read frame from internal camera.")

cap.release()
print("Test complete. Check data/internal_test.jpg if saved.")
