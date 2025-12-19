# main.py
from config import Config
from tracker import HandTracker
from plotter import plot_metrics
from ds import ds


def main():
    tracker = HandTracker(Config.MODEL_PATH, Config.VIDEO_PATH)
    
    print("Processing video...")
    tracker.process_video(show_preview=Config.SHOW_PREVIEW)
    
    metrics, track_id = tracker.get_metrics()
    
    if metrics is None:
        print("No tracked hand found.")
        return

    print(f"Video processing complete. Plotting metrics for track ID: {track_id}")
    print("Preparing data for deepseek to handle the metrics...")
    ds()

    
    print("Displaying metrics plot...")
    plot_metrics(metrics, track_id, figsize=Config.PLOT_FIGSIZE)

if __name__ == "__main__":
    main()