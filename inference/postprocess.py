import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def ema_smooth(scores: List[float], alpha: float = 0.3) -> List[float]:
    """
    Apply exponential moving average smoothing to a list of scores.
    
    Args:
        scores: List of raw anomaly scores.
        alpha: Smoothing factor (0 < alpha <= 1).
        
    Returns:
        List of smoothed scores.
    """
    if not scores:
        return []
    smoothed = [scores[0]]
    for t in range(1, len(scores)):
        smoothed.append(alpha * scores[t] + (1 - alpha) * smoothed[-1])
    return smoothed

def hysteresis_threshold(scores: List[float], enter_threshold: float, exit_delta: float = 0.1, run_length: int = 3) -> List[Tuple[int, int]]:
    """
    Implement hysteresis thresholding with run-length requirements.
    
    Args:
        scores: List of scores (e.g., smoothed scores).
        enter_threshold: Score threshold to trigger an anomaly.
        exit_delta: Score must drop below enter_threshold - exit_delta to end anomaly.
        run_length: Number of consecutive clips required to change state.
        
    Returns:
        List of (start_clip_idx, end_clip_idx) tuples indicating anomaly intervals.
    """
    intervals = []
    state = "NORMAL"
    consecutive = 0
    start_idx = -1

    for i, score in enumerate(scores):
        if state == "NORMAL":
            if score >= enter_threshold:
                consecutive += 1
                if consecutive >= run_length:
                    state = "ANOMALY"
                    start_idx = i - run_length + 1
                    consecutive = 0
            else:
                consecutive = 0
        elif state == "ANOMALY":
            if score < (enter_threshold - exit_delta):
                consecutive += 1
                if consecutive >= run_length:
                    state = "NORMAL"
                    intervals.append((start_idx, i - run_length))
                    consecutive = 0
            else:
                consecutive = 0
    
    if state == "ANOMALY":
        intervals.append((start_idx, len(scores) - 1))
        
    return intervals

def intervals_to_timestamps(intervals: List[Tuple[int, int]], clip_timestamps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Convert clip-index intervals to timestamp intervals.
    
    Args:
        intervals: List of (start_clip_idx, end_clip_idx) tuples.
        clip_timestamps: List of (start_time, end_time) tuples for each clip.
        
    Returns:
        List of (start_time, end_time) tuples indicating anomaly timestamps.
    """
    res = []
    for start, end in intervals:
        start_time = clip_timestamps[start][0]
        end_time = clip_timestamps[end][1]
        res.append((start_time, end_time))
    return res

def generate_score_csv(scores: List[float], smoothed_scores: List[float], timestamps: List[Tuple[float, float]], intervals: List[Tuple[float, float]], output_path: str):
    """
    Write a CSV with columns: clip_idx, start_sec, end_sec, raw_score, smoothed_score, is_anomaly.
    
    Args:
        scores: List of raw scores.
        smoothed_scores: List of smoothed scores.
        timestamps: List of (start_sec, end_sec) for each clip.
        intervals: List of (start_time, end_time) for anomalies.
        output_path: Path to save the CSV.
    """
    is_anomaly = [0] * len(scores)
    
    for i in range(len(scores)):
        clip_start, clip_end = timestamps[i]
        anomaly = 0
        for st, en in intervals:
            # Check for overlap or containment
            if st <= clip_start and clip_end <= en:
                anomaly = 1
                break
        is_anomaly[i] = anomaly
        
    df = pd.DataFrame({
        "clip_idx": range(len(scores)),
        "start_sec": [t[0] for t in timestamps],
        "end_sec": [t[1] for t in timestamps],
        "raw_score": scores,
        "smoothed_score": smoothed_scores,
        "is_anomaly": is_anomaly
    })
    df.to_csv(output_path, index=False)

def plot_score_timeline(scores: List[float], smoothed_scores: List[float], timestamps: List[Tuple[float, float]], intervals: List[Tuple[float, float]], output_path: str, video_name: str = ''):
    """
    Plot score timeline with flagged intervals shaded in red.
    
    Args:
        scores: List of raw scores.
        smoothed_scores: List of smoothed scores.
        timestamps: List of (start_sec, end_sec) for each clip.
        intervals: List of (start_time, end_time) for anomalies.
        output_path: Path to save the PNG image.
        video_name: Name of the video for the title.
    """
    time_axis = [t[0] for t in timestamps]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, scores, label='Raw Score', alpha=0.5, color='gray')
    plt.plot(time_axis, smoothed_scores, label='Smoothed Score', color='blue', linewidth=2)
    
    # Track if we already added the anomaly label to legend
    added_anomaly_label = False
    
    for st, en in intervals:
        if not added_anomaly_label:
            plt.axvspan(st, en, color='red', alpha=0.3, label='Anomaly')
            added_anomaly_label = True
        else:
            plt.axvspan(st, en, color='red', alpha=0.3)
            
    plt.xlabel('Time (s)')
    plt.ylabel('Anomaly Score')
    title = f'Anomaly Score Timeline: {video_name}' if video_name else 'Anomaly Score Timeline'
    plt.title(title)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
