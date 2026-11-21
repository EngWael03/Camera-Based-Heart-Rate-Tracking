from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import shutil
import os

app = FastAPI()

# السماح للـ Frontend (Next.js الذي يعمل على بورت 3000) بالاتصال بالـ Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    # حفظ الفيديو المؤقت المرفوع
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_sec = 60.0
    max_frames = int(max_sec * fps)

    raw_green = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        box = 90
        x1 = (w - box) // 2
        y1 = max(0, ((h - box) // 2) - 120)
        x2, y2 = x1 + box, y1 + box

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        green_ch = roi[:, :, 1]
        skin_idx = np.where(mask == 255)
        skin_pixels = green_ch[skin_idx]

        avg_g = np.mean(skin_pixels) if len(skin_pixels) > 0 else np.mean(green_ch)
        raw_green.append(avg_g)
        frame_count += 1

    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    if len(raw_green) < int(fps * 5):
        return {"error": "Video is too short or no face/skin detected properly."}

    signal = np.array(raw_green)
    low = 0.66 / (fps / 2)
    high = 2.0 / (fps / 2)

    try:
        b, a = butter(N=3, Wn=[low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        min_dist = int(0.5 * fps)
        peaks, _ = find_peaks(filtered_signal, distance=min_dist)

        if len(peaks) < 2:
            return {"bpm": 0.0, "message": "Could not detect enough peaks."}

        peak_diffs = np.diff(peaks)
        avg_diff_sec = np.mean(peak_diffs) / fps
        total_bpm = 60.0 / avg_diff_sec

        return {"bpm": round(float(total_bpm), 1)}
    except Exception as e:
        return {"error": str(e)}