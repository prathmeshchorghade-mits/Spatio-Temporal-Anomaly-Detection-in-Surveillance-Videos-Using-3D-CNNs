# Implementation Plan (Local Execution Scripts)

With the core library components (`training/model`, `training/loss.py`, `training/metrics.py`, etc.) completed, the project requires four top-level execution scripts to wire these modules together. 

These scripts will be run on the target machine where the dataset is physically stored on the hard drive.

---

## 1. `prepare_data.py` (Data Pipeline)
**Objective:** Process the raw UCF-Crime dataset located on the local hard drive into a format ready for training.

**Tasks:**
- Accept the raw dataset directory path via command-line arguments.
- Iterate through all `.mp4` video files.
- Transcode the videos to **10 FPS** at a lower resolution (e.g., 128x171) using OpenCV or FFmpeg to accelerate training I/O.
- Validate that each processed video contains a minimum of 16 frames. Drop corrupted or short videos.
- Generate three dataset manifest CSV files (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`). These files must contain the file paths, labels (normal/anomaly), and categories.

---

## 2. `run_training.py` (Main Training Script)
**Objective:** The entry-point script to initiate the model training process.

**Tasks:**
- Parse command-line arguments (e.g., batch size, epochs, paths to manifests).
- Load the configuration parameters using `ProjectConfig`.
- Instantiate the dataset classes (`MILBagDataset` for training and validation) using the CSV manifests generated in Step 1.
- Instantiate the custom 3D CNN model (`Scratch3DCNN`), the MIL ranking loss function, and the optimizer.
- Invoke the existing main training loop from `training/train.py`.
- Ensure the training loop saves `best_model.pt` and `last_model.pt` correctly to the `./checkpoints/` directory.

---

## 3. `run_evaluation.py` (Evaluation Script)
**Objective:** Test the model's performance after training and generate standardized metrics.

**Tasks:**
- Load the frozen `best_model.pt` checkpoint.
- Run inference over the **validation set** to calibrate and lock in the optimal anomaly threshold.
- Run inference over the **test set** using the calibrated threshold.
- Utilize the functions in `training/metrics.py` to output the final frame-level ROC-AUC, PR-AUC, and false alarm rates.
- Generate failure analysis logs (e.g., flagging low light or motion blur).

---

## 4. `predict_video.py` (Inference & Demo Script)
**Objective:** Demonstrate the model's capabilities on a single, unseen surveillance video.

**Tasks:**
- Accept a path to a single `.mp4` video as input.
- Extract 16-frame overlapping clips and run them through the trained model to get raw anomaly scores.
- Apply temporal smoothing to the scores using the utilities in `inference/postprocess.py`.
- Use OpenCV to draw a score overlay/graph and colored bounding box alerts directly onto the video frames.
- Save the annotated `.mp4` output file to the `./outputs/` directory for human review.
