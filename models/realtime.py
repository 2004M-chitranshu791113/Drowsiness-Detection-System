import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import pygame

# ================== LOAD MODEL ==================
model = load_model("models/drowsy_model.h5")

# ================== MEDIAPIPE ================== 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

closed_start = None
alarm_on = False
frame_count = 0

# ================== BUZZER (FIXED) ==================
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # loop

def stop_alarm():
    pygame.mixer.music.stop()

# ================== ROI EXTRACTION ==================
def extract_roi(frame, landmarks):
    h, w, _ = frame.shape

    left_eye = [33, 133]
    right_eye = [362, 263]

    points = left_eye + right_eye

    coords = [(int(landmarks.landmark[p].x*w),
               int(landmarks.landmark[p].y*h)) for p in points]

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    x1, x2 = min(xs)-20, max(xs)+20
    y1, y2 = min(ys)-20, max(ys)+20

    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1, x2, y2

# ================== GRAD-CAM ==================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# ================== APPLY HEATMAP ==================
def apply_gradcam(frame, heatmap, x1, y1, x2, y2):
    heatmap = cv2.resize(heatmap, (x2 - x1, y2 - y1))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    frame[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 0.6,
        heatmap, 0.4, 0
    )

    return frame

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        for face in res.multi_face_landmarks:

            roi, x1, y1, x2, y2 = extract_roi(frame, face)

            if roi is None or roi.size == 0:
                continue

            roi = cv2.resize(roi, (112,112))
            roi_input = roi / 255.0
            roi_input = np.expand_dims(roi_input, axis=0)

            pred = model.predict(roi_input, verbose=0)[0][1]

            # ===== SHOW PROBABILITY =====
            cv2.putText(frame, f"Prob: {pred:.2f}", (50,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # ===== SHOW STATE =====
            if pred > 0.7:
                state = "DROWSY"
                color = (0,0,255)
            else:
                state = "AWAKE"
                color = (0,255,0)

            cv2.putText(frame, state, (50,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # ===== Grad-CAM every 5 frames =====
            if frame_count % 5 == 0:
                try:
                    heatmap = get_gradcam_heatmap(model, roi_input)
                    frame = apply_gradcam(frame, heatmap, x1, y1, x2, y2)
                except:
                    pass

            # ===== DROWSINESS LOGIC =====
            if pred > 0.75:
                if closed_start is None:
                    closed_start = time.time()
                else:
                    elapsed = (time.time() - closed_start) * 1000

                    if elapsed > 300:
                        cv2.putText(frame, "DROWSY ALERT!", (50,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

                        if not alarm_on:
                            play_alarm()
                            alarm_on = True
            else:
                closed_start = None

                if alarm_on:
                    stop_alarm()
                    alarm_on = False

    frame_count += 1

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()