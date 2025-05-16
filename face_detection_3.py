import cv2
import numpy as np
from deepface import DeepFace

# Configuration
image_path = "foure_face.jpg"
detector_backend = 'mtcnn'  # You can also try 'retinaface' or 'opencv'

# Read and resize image if too large
img = cv2.imread(image_path)
h_img, w_img = img.shape[:2]
if max(h_img, w_img) > 800:
    scale = 800 / max(h_img, w_img)
    img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))

# Preprocessing: Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Convert to YUV and apply histogram equalization on the Y channel
img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Run analysis with DeepFace
results = DeepFace.analyze(
    img_path=img_eq,
    actions=['emotion'],
    detector_backend=detector_backend,
    enforce_detection=False
)

# Normalize results
if not isinstance(results, list):
    results = [results]

face_id = 0

# Iterate over detected faces
for face in results:
    region = face.get("region", {})
    if not region:
        continue

    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    emotion = face.get("dominant_emotion", "unknown")

    # Filter 1: size
    if w < 50 or h < 50 or w > 350 or h > 350:
        continue

    # Filter 2: aspect ratio
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.6 or aspect_ratio > 1.5:
        continue

    # Draw rectangle and label
    face_id += 1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(f"face_{face_id}.jpg", cropped)

# Show result
cv2.imshow("Filtered Faces with Emotions", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

