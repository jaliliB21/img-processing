import cv2
import dlib
import numpy as np
from deepface import DeepFace

# Load face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# Load and optionally resize image
image_path = "foure_face.jpg"
img = cv2.imread(image_path)
orig_img = img.copy()
h, w = img.shape[:2]
if max(h, w) > 800:
    scale = 800 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

# Detect faces
faces = detector(img)

for i, face in enumerate(faces):
    landmarks = predictor(img, face)

    # Get eye regions (landmark points)
    left_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
    right_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

    # Bounding rectangles for eyes
    def get_eye_color(eye_pts):
        x, y, w, h = cv2.boundingRect(eye_pts)
        eye_roi = img[y:y+h, x:x+w]
        if eye_roi.size == 0:
            return "unknown"
        avg_color = eye_roi.mean(axis=0).mean(axis=0)
        b, g, r = avg_color
        if r > 100 and g > 80 and b > 80:
            return "hazel"
        elif g > 100 and r < 90:
            return "green"
        elif b > 100 and g < 80:
            return "blue"
        elif r < 80 and g < 80 and b < 80:
            return "black"
        else:
            return "brown"

    left_color = get_eye_color(left_eye_pts)
    right_color = get_eye_color(right_eye_pts)

    # Average both eye colors
    final_color = left_color if left_color == right_color else f"{left_color}/{right_color}"

    # Emotion detection using DeepFace on cropped face
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_crop = img[y1:y2, x1:x2]
    emotion = "unknown"
    try:
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    except:
        pass

    # Draw face box and label number
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)

    # Print to terminal
    print(f"[Face {i+1}] Emotion: {emotion} | Eye Color: {final_color}")

# Show image
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

