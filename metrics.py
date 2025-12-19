# metrics.py
import numpy as np

def compute_frame_metrics(wrist_history):
    """
    根据手腕历史位置列表，计算当前帧的运动指标。
    param wrist_history: list of (x, y), at least length 2
    return: dict of metric values
    """
    if len(wrist_history) < 2:
        return {"euclid": 0.0, "cos": 1.0, "angle": 0.0}
    
    x, y = wrist_history[-1]
    x_prev, y_prev = wrist_history[-2]
    dx = x - x_prev
    dy = y - y_prev
    euclid = np.sqrt(dx**2 + dy**2)

    if len(wrist_history) < 3:
        return {"euclid": euclid, "cos": 1.0, "angle": 0.0}

    x_prev2, y_prev2 = wrist_history[-3]
    v1 = np.array([x_prev - x_prev2, y_prev - y_prev2])
    v2 = np.array([dx, dy])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        cos_sim, angle_deg = 1.0, 0.0
    else:
        cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_sim))

    return {
        "euclid": euclid,
        "cos": cos_sim,
        "angle": angle_deg
    }