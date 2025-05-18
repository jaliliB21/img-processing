import cv2
import numpy as np
from deepface import DeepFace

# Eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Image setup
image_path = "face_5.jpg"
detector_backend = 'mtcnn'

img = cv2.imread(image_path)
h_img, w_img = img.shape[:2]
if max(h_img, w_img) > 800:
    scale = 800 / max(h_img, w_img)
    img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))

# Preprocessing
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# DeepFace analysis
results = DeepFace.analyze(
    img_path=img_eq,
    actions=['emotion'],
    detector_backend=detector_backend,
    enforce_detection=False
)

if not isinstance(results, list):
    results = [results]

face_id = 0

def get_eye_color(eye_roi):
    # Crop center to avoid eyelids/brows
    h, w = eye_roi.shape[:2]
    cx, cy = w // 4, h // 4
    cropped = eye_roi[cy:cy*3, cx:cx*3]  # center region

    avg_color = np.mean(cropped.reshape(-1, 3), axis=0)
    b, g, r = avg_color

    # Simple heuristic
    if r < 70 and g < 70 and b < 70:
        return "black"
    elif b > 120 and b > r and b > g:
        return "blue"
    elif g > 90 and r > 80 and g > b:
        return "green"
    elif r > 90 and g > 60:
        return "brown"
    else:
        return "unknown"

# Process faces
for face in results:
    region = face.get("region", {})
    if not region:
        continue

    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    emotion = face.get("dominant_emotion", "unknown")

    # Filtering
    if w < 50 or h < 50 or w > 350 or h > 350:
        continue
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.6 or aspect_ratio > 1.5:
        continue

    face_crop = img[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

    eye_colors = []
    for (ex, ey, ew, eh) in eyes[:2]:  # max 2 eyes
        eye_roi = face_crop[ey:ey+eh, ex:ex+ew]
        eye_color = get_eye_color(eye_roi)
        eye_colors.append(eye_color)

    final_eye_color = eye_colors[0] if eye_colors else "unknown"

    face_id += 1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Console output
    print(f"Face {face_id}: Emotion = {emotion}, Eye Color = {final_eye_color}")

    # Optional: Save cropped faces
    cv2.imwrite(f"face_{face_id}.jpg", face_crop)

# Show result
cv2.imshow("Faces with Indexes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

