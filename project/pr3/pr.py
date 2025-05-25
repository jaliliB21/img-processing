import cv2
import dlib
import numpy as np
from deepface import DeepFace

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(0)

def get_eye_color(img, eye_pts):
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Problem receiving the image from the camera.")
        break

    img = frame.copy()
    faces = detector(img)

    for i, face in enumerate(faces):
        landmarks = predictor(img, face)

        left_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
        right_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

        left_color = get_eye_color(img, left_eye_pts)
        right_color = get_eye_color(img, right_eye_pts)
        final_color = left_color if left_color == right_color else f"{left_color}/{right_color}"

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_crop = img[y1:y2, x1:x2]

        emotion = "unknown"
        try:
            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        except Exception as e:
            print("Error in recognizing emotion:", e)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Face {i+1}: {emotion} | {final_color}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    cv2.imshow("Live Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

