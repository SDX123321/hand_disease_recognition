# config.py

class Config:
    # Model
    MODEL_PATH = "./best.pt"
    
    # Video
    VIDEO_PATH = "/root/yolo/DJI_20251217105958_0174_D.MP4"
    
    # Tracking
    MAX_HISTORY_FRAMES = 30
    TARGET_METRICS = ["euclid", "cos", "angle"]
    
    # Visualization
    SHOW_PREVIEW = True
    PREVIEW_WINDOW_NAME = "Hand Tracking (Press 'q' to quit)"
    
    # Plotting
    PLOT_FIGSIZE = (12, 9)
    PLOT_DPI = 100