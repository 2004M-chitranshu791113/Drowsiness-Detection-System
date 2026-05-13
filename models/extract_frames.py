import cv2
import os

video_folder = "videos"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

for video in os.listdir(video_folder):
    cap = cv2.VideoCapture(os.path.join(video_folder, video))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(f"{output_folder}/{video}_{count}.jpg", frame)
        count += 1

    cap.release()

print("Frames extracted!")