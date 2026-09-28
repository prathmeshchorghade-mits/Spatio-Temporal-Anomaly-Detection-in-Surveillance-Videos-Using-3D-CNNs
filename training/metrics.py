import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve

def compute_video_level_auc(video_scores: List[float], video_labels: List[int]) -> float:
    """
    Computes the ROC-AUC at the video level.
    
    Args:
        video_scores: List of maximum anomaly scores per video.
        video_labels: List of ground truth labels per video (0 or 1).
        
    Returns:
        ROC-AUC score.
    """
    return roc_auc_score(video_labels, video_scores)

def compute_frame_level_auc(all_scores: np.ndarray, all_labels: np.ndarray) -> Dict[str, float]:
    """
    Computes ROC-AUC and PR-AUC at the frame or clip level.
    
    Args:
        all_scores: Array of concatenated frame/clip-level anomaly scores.
        all_labels: Array of concatenated frame/clip-level ground truth labels.
        
    Returns:
        Dictionary containing 'roc_auc' and 'pr_auc'.
    """
    roc_auc = roc_auc_score(all_labels, all_scores)
    pr_auc = average_precision_score(all_labels, all_scores)
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

def compute_false_alarm_rate(normal_video_scores: List[List[float]], threshold: float, fps: int = 10, clip_stride: int = 8) -> float:
    """
    Computes the false alarm rate per video-hour on normal videos.
    
    Args:
        normal_video_scores: List of lists containing clip/frame anomaly scores for normal videos.
        threshold: Anomaly score threshold to trigger an alarm.
        fps: Frames per second of the video.
        clip_stride: Stride of clips used for scoring.
        
    Returns:
        False alarms per video-hour.
    """
    total_frames = sum(len(scores) for scores in normal_video_scores) * clip_stride
    total_hours = total_frames / (fps * 3600)
    
    # A false alarm is a continuous sequence of scores above threshold
    false_alarms = 0
    for scores in normal_video_scores:
        in_alarm = False
        for score in scores:
            if score >= threshold:
                if not in_alarm:
                    false_alarms += 1
                    in_alarm = True
            else:
                in_alarm = False
                
    if total_hours == 0:
        return 0.0
    return false_alarms / total_hours

def calibrate_threshold(val_scores: List[float], val_labels: List[int], target_fpr: float = 0.05) -> float:
    """
    Finds the threshold on validation set that achieves a target False Positive Rate.
    
    Args:
        val_scores: List of scores for validation set.
        val_labels: List of ground truth labels for validation set.
        target_fpr: Target False Positive Rate.
        
    Returns:
        Threshold value.
    """
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    # Find the largest threshold such that fpr <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return thresholds[0]
    best_idx = valid_indices[-1]
    return float(thresholds[best_idx])

def compute_category_breakdown(video_scores: List[float], video_labels: List[int], video_categories: List[str], threshold: float) -> pd.DataFrame:
    """
    Computes evaluation metrics broken down by video category.
    
    Args:
        video_scores: List of anomaly scores per video.
        video_labels: List of ground truth labels per video.
        video_categories: List of category strings per video.
        threshold: Score threshold for binary predictions.
        
    Returns:
        DataFrame containing metrics per category.
    """
    df = pd.DataFrame({
        'score': video_scores,
        'label': video_labels,
        'category': video_categories,
        'pred': [1 if s >= threshold else 0 for s in video_scores]
    })
    
    records = []
    for cat, group in df.groupby('category'):
        count = len(group)
        if len(group['label'].unique()) > 1:
            auc = roc_auc_score(group['label'], group['score'])
        else:
            auc = np.nan
            
        # For precision, recall, f1, we use zero_division=0 to prevent warnings
        precision, recall, f1, _ = precision_recall_fscore_support(
            group['label'], group['pred'], average='binary', zero_division=0
        )
        
        records.append({
            'category': cat,
            'count': count,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
    return pd.DataFrame(records)

def plot_roc_pr_curves(scores: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Plots and saves ROC and Precision-Recall curves.
    
    Args:
        scores: Array of anomaly scores.
        labels: Array of ground truth labels.
        save_path: Path to save the resulting figure.
    """
    from sklearn.metrics import precision_recall_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    # PR Curve
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
