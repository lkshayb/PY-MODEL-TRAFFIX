import time
from collections import defaultdict
import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO
from multiprocessing import Process, Manager
import base64

app = Flask(__name__, template_folder="../templates")

print(f"CUDA DEVICE IN USE ==>{torch.cuda.get_device_name(0)}")
torch.backends.cudnn.benchmark = True

#Model Setup
model = YOLO("yolo11n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
TRACKER_CFG = "bytetrack.yaml"


video_paths = ["./v2.mp4", "./v4.mp4", "./v3.mp4"]
roi_list = [
    np.array([[100, 150], [1200, 150], [1200, 650], [100, 650]], np.int32),
    np.array([[50, 100], [1100, 100], [1100, 600], [50, 600]], np.int32),
    np.array([[200, 200], [1000, 200], [1000, 700], [200, 700]], np.int32)
]

def in_roi(pt, roi):
    return cv2.pointPolygonTest(roi, pt, False) >= 0

# ===================== Vehicle Map =====================
vehicle_map = {2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck", 1: "Bicycle"}
colors = {
    "Car": (0, 200, 0),
    "Motorbike": (0, 255, 255),
    "Bus": (255, 80, 0),
    "Truck": (0, 80, 255),
    "Bicycle": (180, 180, 0),
}

# ===================== Traffic Signal Controller =====================
class TrafficSignal:
    DEFAULT_RED = 120
    DEFAULT_GREEN = 60
    MIN_GREEN = 10
    MAX_GREEN = 90
    YELLOW = 4

    def __init__(self):
        self.state = "RED"
        self.start_time = time.time()
        self.duration = self.DEFAULT_RED

    def decide_timing(self, vehicle_count: int) -> int:
        if vehicle_count == 0:
            return self.MIN_GREEN
        elif vehicle_count < 5:
            return 20
        elif vehicle_count < 15:
            return 40
        elif vehicle_count < 30:
            return 60
        else:
            return self.DEFAULT_GREEN

    def update(self, vehicle_count: int):
        now = time.time()
        elapsed = now - self.start_time

        if self.state == "RED" and elapsed >= self.duration:
            self.state = "GREEN"
            self.start_time = now
            self.duration = self.decide_timing(vehicle_count)

        elif self.state == "GREEN" and elapsed >= self.duration:
            self.state = "YELLOW"
            self.start_time = now
            self.duration = self.YELLOW

        elif self.state == "YELLOW" and elapsed >= self.duration:
            self.state = "RED"
            self.start_time = now
            self.duration = self.DEFAULT_RED

    def get_status(self):
        elapsed = time.time() - self.start_time
        return self.state, max(0, int(self.duration - elapsed))

# ===================== Worker Function =====================
def video_worker(i, video_path, roi, shared_data):
    cap = cv2.VideoCapture(video_path)
    signal = TrafficSignal()

    while True:
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(video_path)
            continue

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        with torch.no_grad():
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
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)

                if in_roi((cx, cy), roi):
                    counts[name] += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[name], 2)
                    cv2.putText(frame, name, (int(x1), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)

        # ROI polygon
        cv2.polylines(frame, [roi], True, (0, 255, 255), 3)

        # Update signal
        total_vehicles = sum(counts.values())
        signal.update(total_vehicles)
        state, time_left = signal.get_status()

        # Overlay signal
        color = (0, 0, 255) if state == "RED" else (0, 255, 0) if state == "GREEN" else (0, 255, 255)
        cv2.putText(frame, f"{state} ({time_left}s)", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Update shared stats
        shared_data[i]["stats"] = {"vehicles": dict(counts), "signal": state, "time_left": time_left}

        # Encode frame -> base64 for sharing
        _, buf = cv2.imencode(".jpg", frame)
        shared_data[i]["frame"] = base64.b64encode(buf).decode("utf-8")

# ===================== Flask Routes =====================
@app.route("/")
def index():
    return render_template("index.html")

def frame_generator(i, shared_data):
    while True:
        if "frame" in shared_data[i]:
            jpg_bytes = base64.b64decode(shared_data[i]["frame"])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        else:
            time.sleep(0.05)

@app.route("/video_feed<int:i>")
def video_feed(i):
    return Response(frame_generator(i-1, shared_data),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats<int:i>")
def stats(i):
    return jsonify(shared_data[i-1].get("stats", {}))

# ===================== Main =====================
if __name__ == "__main__":
    manager = Manager()
    shared_data = manager.list([manager.dict() for _ in range(len(video_paths))])

    workers = []
    for i, path in enumerate(video_paths):
        p = Process(target=video_worker, args=(i, path, roi_list[i], shared_data))
        p.start()
        workers.append(p)

    app.run(debug=True, threaded=True)
