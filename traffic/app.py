import time
from collections import  defaultdict
import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__, template_folder="../templates")



# ===== CUDA Setup =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"Using CUDA with: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    print("CUDA not available, using CPU")

# ===== Model & Video Paths =====
model = YOLO("yolo11n.pt")
model.to(device)
TRACKER_CFG = "bytetrack.yaml"
video_paths = [r"./v1.mp4",r"./v2.mp4",r"./v3.mp4"]
caps = [cv2.VideoCapture(v) for v in video_paths]

# ==== ROI (rectangular, can adjust per video) ====
roi = np.array([[100, 150], [1200, 150], [1200, 650], [100, 650]], np.int32)

# ==== Vehicle map & colors ====
vehicle_map = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck", 1: "Bicycle"}
colors = {"Car": (0, 200, 0), "Motorbike": (0, 255, 255),"Bus": (255, 80, 0), "Truck": (0, 80, 255), "Bicycle": (180, 180, 0)}

# ==== Traffic control params ====
DEFAULT_RED = 120
DEFAULT_GREEN = 60
MIN_GREEN = 10
MAX_GREEN = 90
YELLOW = 4

# ==== State tracking ====
states = ["RED"] * 3
state_starts = [time.time()] * 3
durations = [DEFAULT_RED] * 3
# Always keep a well-formed stats object to avoid frontend null checks
last_stats = [
    {
        "vehicles": {},
        "signal": "RED",
        "time_left": DEFAULT_RED
    }
    for _ in range(3)
]


def in_roi(pt): return cv2.pointPolygonTest(roi, pt, False) >= 0


def decide_timing(vehicle_count):
    """Adaptive green time based on traffic density."""
    if vehicle_count == 0:
        return MIN_GREEN
    if vehicle_count < 5:
        return 20
    elif vehicle_count < 15:
        return 40
    elif vehicle_count < 30:
        return 60
    else:
        return DEFAULT_GREEN


def update_signal(i, vehicle_count):
    now = time.time()
    elapsed = now - state_starts[i]

    if states[i] == "RED":
        if elapsed >= durations[i]:
            states[i] = "GREEN"
            state_starts[i] = now
            durations[i] = decide_timing(vehicle_count)

    elif states[i] == "GREEN":
        if elapsed >= durations[i]:
            states[i] = "YELLOW"
            state_starts[i] = now
            durations[i] = YELLOW

    elif states[i] == "YELLOW":
        if elapsed >= durations[i]:
            states[i] = "RED"
            state_starts[i] = now
            durations[i] = DEFAULT_RED


def generate_frames(i):
    cap = caps[i]
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run tracker (model accepts numpy images directly). Keep a single code path
        results = model.track(frame, tracker=TRACKER_CFG, persist=True, verbose=False)
        
        counts = defaultdict(int)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().tolist()
            clss = results[0].boxes.cls.int().tolist()
            xyxys = results[0].boxes.xyxy.tolist()

            for tid, cls, (x1, y1, x2, y2) in zip(ids, clss, xyxys):
                if cls not in vehicle_map:
                    continue
                name = vehicle_map[cls]
                cx, cy = (int((x1 + x2) // 2), int((y1 + y2) // 2))

                if in_roi((cx, cy)):
                    counts[name] += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[name], 2)
                    cv2.putText(frame, name, (int(x1), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)

        # ROI boundary
        cv2.polylines(frame, [roi], True, (0, 255, 255), 3)

        total_vehicles = sum(counts.values())
        update_signal(i, total_vehicles)

        elapsed = time.time() - state_starts[i]
        time_left = int(durations[i] - elapsed)

        # Draw signal status
        color = (0, 0, 255) if states[i] == "RED" else (0, 255, 0) if states[i] == "GREEN" else (0, 255, 255)
        cv2.putText(frame, f"{states[i]} ({time_left}s)", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        last_stats[i] = {
            "vehicles": dict(counts),
            "signal": states[i],
            "time_left": max(0, time_left)
        }

        _, buf = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")

# Register routes for each feed
for i in range(3):
    app.add_url_rule(f"/video_feed{i+1}", f"video_feed{i+1}",
                     lambda i=i: Response(generate_frames(i), mimetype="multipart/x-mixed-replace; boundary=frame"))
    app.add_url_rule(f"/stats{i+1}", f"stats{i+1}", lambda i=i: jsonify(last_stats[i]))

if __name__ == "__main__":
    app.run(debug=True)
