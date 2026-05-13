import cv2
import mediapipe as mp
import os
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

input_folder = "frames"
output_drowsy = "dataset/drowsy"
output_alert = "dataset/not_drowsy"

os.makedirs(output_drowsy, exist_ok=True)
os.makedirs(output_alert, exist_ok=True)

def extract_eye_roi(frame, landmarks):
    h, w, _ = frame.shape

    points = [63, 117, 293, 346, 9]

    coords = []
    for p in points:
        lm = landmarks.landmark[p]
        coords.append((int(lm.x * w), int(lm.y * h)))

    (x1, y1), (_, _), (_, _), (x2, y2), (xm, ym) = coords

    # ROI correction logic (simplified)
    start_x = min(x1, x2)
    end_x = max(x1, x2)
    start_y = min(y1, y2)
    end_y = max(y1, y2)

    roi = frame[start_y:end_y, start_x:end_x]

    return roi


count_d = 0
count_nd = 0

for img in os.listdir(input_folder):
    frame = cv2.imread(os.path.join(input_folder, img))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            roi = extract_eye_roi(frame, landmarks)

            if roi is None or roi.size == 0:
                continue

            roi = cv2.resize(roi, (112, 112))

            cv2.imshow("ROI", roi)
            key = cv2.waitKey(0)

            if key == ord('d'):
                cv2.imwrite(f"{output_drowsy}/{count_d}.jpg", roi)
                count_d += 1
            else:
                cv2.imwrite(f"{output_alert}/{count_nd}.jpg", roi)
                count_nd += 1

cv2.destroyAllWindows()