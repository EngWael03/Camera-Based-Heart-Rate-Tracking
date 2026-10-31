import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

video_path = r"inputfileDirectory.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

max_sec = 60.0
max_frames = int(max_sec * fps)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames > max_frames:
    total_frames = max_frames

total_sec = total_frames / fps

print("FPS:", fps)
print("Frames to process:", total_frames)

raw_green = []
frame_count = 0

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    h = frame.shape[0]
    w = frame.shape[1]

    box = 90
    x1 = (w - box) // 2

    y1 = ((h - box) // 2) - 120
    if y1 < 0:
        y1 = 0

    x2 = x1 + box
    y2 = y1 + box

    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    green_ch = roi[:, :, 1]

    skin_idx = np.where(mask == 255)
    skin_pixels = green_ch[skin_idx]

    if len(skin_pixels) > 0:
        avg_g = np.mean(skin_pixels)
    else:
        avg_g = np.mean(green_ch)

    raw_green.append(avg_g)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Video", frame)

    frame_count += 1
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

signal = np.array(raw_green)

low = 0.66 / (fps / 2)
high = 2.0 / (fps / 2)

b, a = butter(N=3, Wn=[low, high], btype='band')
filtered_signal = filtfilt(b, a, signal)

min_dist = int(0.5 * fps)
peaks, _ = find_peaks(filtered_signal, distance=min_dist)

peak_diffs = np.diff(peaks)
avg_diff_sec = np.mean(peak_diffs) / fps
total_bpm = 60.0 / avg_diff_sec

print("\n" + "=" * 40)
print(f"OVERALL HEART RATE: {total_bpm:.1f} BPM")
print("=" * 40)

print("\n--- 10s Intervals ---")
frames_10s = int(10 * fps)
interval_num = 1

for i in range(0, len(filtered_signal), frames_10s):
    sub_signal = filtered_signal[i: i + frames_10s]

    if len(sub_signal) < (5 * fps):
        continue

    local_peaks, _ = find_peaks(sub_signal, distance=min_dist)

    if len(local_peaks) >= 2:
        local_diffs = np.diff(local_peaks)
        local_avg_sec = np.mean(local_diffs) / fps
        local_bpm = 60.0 / local_avg_sec

        t1 = (interval_num * 10) - 10
        t2 = min(interval_num * 10, total_sec)
        print(f"Interval {interval_num} ({t1}s - {t2:.1f}s): {local_bpm:.1f} BPM")

    interval_num += 1

print("\n--- 1s Intervals ---")

peak_times = peaks / fps
bpm_list = []

for i in range(len(peak_diffs)):
    diff_in_seconds = peak_diffs[i] / fps
    instant_bpm = 60.0 / diff_in_seconds
    bpm_list.append(instant_bpm)

for sec in range(int(total_sec)):
    start = sec
    end = sec + 1

    valid_bpms = []
    for i in range(len(peak_times)):
        t = peak_times[i]
        if t >= start and t < end:
            if i > 0:
                valid_bpms.append(bpm_list[i - 1])

    if len(valid_bpms) > 0:
        avg_1s_bpm = np.mean(valid_bpms)
        print(f"Second {start:02d} to {end:02d}: {avg_1s_bpm:.1f} BPM")
    else:
        print(f"Second {start:02d} to {end:02d}: No peak in this window")

t_axis = np.linspace(0, total_sec, len(filtered_signal))

plt.figure(figsize=(10, 5))
plt.plot(t_axis, filtered_signal, color='green', label="Filtered Signal")
plt.plot(t_axis[peaks], filtered_signal[peaks], "ro", label=f"Beats ({total_bpm:.1f} BPM)")
plt.title("rPPG Heart Rate Tracker")
plt.xlabel("Time (seconds)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.show()
