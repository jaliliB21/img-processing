import cv2
from deepface import DeepFace

# Use a more reliable detector: 'opencv' or 'mtcnn'
detector_backend = 'mtcnn'  # or 'opencv'


image_path = "two_face.jpg"

# Read and resize image if too large
img = cv2.imread(image_path)

# Analyze
results = DeepFace.analyze(
    img_path=image_path,
    actions=['emotion'],
    detector_backend=detector_backend,
    enforce_detection=False
)

# Normalize results
if not isinstance(results, list):
    results = [results]

face_id = 0

for face in results:
    region = face.get("region", {})
    if not region:
        continue
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    emotion = face.get("dominant_emotion", "unknown")

    # Filter 1: small/large faces
    if w < 50 or h < 50 or w > 300 or h > 300:
        continue

    # Filter 2: weird aspect ratio
    aspect_ratio = w / float(h)
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        continue

    # If passed filters
    face_id += 1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(f"face_{face_id}.jpg", cropped)

resized_img = cv2.resize(img, (600, 800))
cv2.imshow("Faces with Emotions", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

