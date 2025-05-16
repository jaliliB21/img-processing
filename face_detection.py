import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load the image
image_path = "two_face.jpg"
img = cv2.imread(image_path)


# Analyze image: detect faces and emotions
results = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

# Check if multiple faces or single face detected
if not isinstance(results, list):
    results = [results]

# Loop through detected faces
for i, face in enumerate(results):
    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
    emotion = face['dominant_emotion']

    # Draw rectangle and emotion label
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    # Crop the face for later use (e.g. eye color detection)
    face_crop = img[y:y+h, x:x+w]
    cv2.imwrite(f"face_{i+1}.jpg", face_crop)

# Show the result


resized_img = cv2.resize(img, (600, 800))
cv2.imshow("Faces with Emotions", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

