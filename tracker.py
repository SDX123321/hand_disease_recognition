# tracker.py
from collections import defaultdict
from ultralytics import YOLO
from config import Config
from metrics import compute_frame_metrics
import cv2
import numpy as np
import os
from datetime import datetime
import json

class HandTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.wrist_history = defaultdict(list)
        self.metrics_history = defaultdict(lambda: {
            "euclid": [], "cos": [], "angle": []
        })
        self.target_track_id = None
        self.CHART_W = 240
        self.CHART_H = 120
        self.MARGIN = 10
        self.COLORS = {
            "euclid": (0, 255, 255),    # Yellow
            "cos": (255, 0, 255),       # Magenta
            "angle": (0, 165, 255)      # Orange
        }
        self.METRICS = ["euclid", "cos", "angle"]

        self.history_dir='history'
        # os.mkdir(self.history_dir)

    def _draw_metrics_chart(self, frame, track_id):
        # three pics
        h, w = frame.shape[:2]
        chart_x0 = w - self.CHART_W - self.MARGIN
        chart_y0 = self.MARGIN

        
        if chart_x0 < 0:
            chart_x0 = 0
        if chart_y0 < 0:
            chart_y0 = 0

        
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (chart_x0, chart_y0),
            (chart_x0 + self.CHART_W, chart_y0 + self.CHART_H),
            (20, 20, 20),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        sub_h = self.CHART_H // len(self.METRICS)
        metrics = self.metrics_history[track_id]

        for i, metric in enumerate(self.METRICS):
            y0 = chart_y0 + i * sub_h
            values = metrics[metric]
            if len(values) < 2:
                
                cv2.putText(
                    frame,
                    f"{metric}: {values[-1]:.2f}" if values else f"{metric}: N/A",
                    (chart_x0 + 5, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )
                continue

           # norm
            min_v, max_v = min(values), max(values)
            range_v = max_v - min_v if max_v != min_v else 1.0

            points = []
            for j, val in enumerate(values):
                px = chart_x0 + int(j * (self.CHART_W - 10) / (len(values) - 1)) if len(values) > 1 else chart_x0 + self.CHART_W // 2
                py = y0 + sub_h - 10 - int((val - min_v) / range_v * (sub_h - 20))
                points.append([px, py])

            if len(points) > 1:
                pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_np], False, self.COLORS[metric], 2)

            cv2.putText(
                frame,
                f"{metric}: {values[-1]:.2f}",
                (chart_x0 + 5, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

    def process_video(self, show_preview=True):
        frame_idx = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model.track(frame, persist=True, verbose=False)
            if results:
                result = results[0]
                if result.boxes and result.boxes.is_track:
                    frame = result.plot()
                    pts = result.keypoints.xy.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Record first track_id
                    if self.target_track_id is None and track_ids:
                        self.target_track_id = track_ids[0]

                    for kpts, tid in zip(pts, track_ids):
                        wrist_pt = kpts[0]
                        x, y = float(wrist_pt[0]), float(wrist_pt[1])
                        self.wrist_history[tid].append((x, y))

                        # Limit history length
                        if len(self.wrist_history[tid]) > Config.MAX_HISTORY_FRAMES:
                            self.wrist_history[tid].pop(0)

                        # Compute and store metrics
                        metrics = compute_frame_metrics(self.wrist_history[tid])
                        for key, val in metrics.items():
                            self.metrics_history[tid][key].append(val)
                        if show_preview:
                            self._draw_metrics_chart(frame, tid)

            if show_preview:
                cv2.imshow(Config.PREVIEW_WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if show_preview:
                cv2.imshow(Config.PREVIEW_WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()
        self._save_metrics_to_json()

    def _save_metrics_to_json(self):
        """将所有 track_id 的 metrics 保存为 JSON 文件"""
        if not self.metrics_history:
            print("No metrics to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for track_id, metrics in self.metrics_history.items():
            # 检查是否有有效数据
            if not any(metrics.values()):
                continue

            filename = f"{timestamp}_track{track_id}.json"
            filepath = os.path.join(self.history_dir, filename)

            # 转为标准 dict（确保 JSON 序列化）
            serializable_metrics = {
                key: [float(v) for v in val]  # 转为 Python float
                for key, val in metrics.items()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2)

            print(f"Metrics saved to: {filepath}")

    def get_metrics(self, track_id=None):
        if track_id is None:
            track_id = self.target_track_id
        return self.metrics_history.get(track_id, None), track_id