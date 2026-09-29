# Spatio-Temporal Video Anomaly Detection: Training Guide

This guide provides step-by-step instructions for setting up the environment, placing the raw data, and executing the training pipeline on a Windows machine with an NVIDIA RTX GPU.

## 1. Environment Setup

Before starting, ensure you have Python 3.9+ installed on your Windows machine.

Open your terminal or command prompt in the project root directory and install the required dependencies. The `requirements.txt` is configured to pull the CUDA 12.1 PyTorch binaries for your RTX GPU.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Ensure the output directories exist (they should have been created, but if not, create them):
```bash
mkdir data checkpoints runs outputs
```

## 2. Dataset Preparation

The model trains on the UCF-Crime dataset. You need the raw `.mp4` video files to start.

### Where to put the data
You can place the raw dataset anywhere on your hard drive, but keeping it organized is recommended. For example, place the raw dataset folder at `C:\Datasets\UCF-Crime`. 

The raw dataset folder should ideally contain the video files categorized by anomaly (e.g., `Abuse`, `Arrest`, `Normal`, etc.).

### Run the Data Pipeline
Before you can train the model, you must process these raw videos into 10 FPS transcoded clips and generate the CSV manifests (`manifest_train.csv` and `manifest_val.csv`) that the DataLoader expects.

Run the `prepare_data.py` script. Replace `C:\Datasets\UCF-Crime` with the actual path to your raw video files:

```bash
python prepare_data.py --raw-dir "C:\Datasets\UCF-Crime" --processed-dir ./data --manifest-dir ./data
```

**What this does:**
- Scans your raw directory for `.mp4` files.
- Transcodes them to the target FPS and resolution, saving them into the `./data` folder.
- Generates `manifest_train.csv` and `manifest_val.csv` in the `./data` folder, mapping video paths to labels.

## 3. Start Training

Once the data preparation is complete and the manifests are in the `./data` folder, you are ready to train the 3D CNN model. 

Instead of running a Jupyter Notebook (which can crash or disconnect during long training jobs), a Python script has been created to execute the training loop over 3 different random seeds (42, 123, 456) for robust evaluation.

Start the training run with:

```bash
python train_runner.py
```

### What to expect during training:
- **Checkpoints**: The model will save checkpoints in the `./checkpoints` directory (`best_model_seed42.pt`, `last_model_seed42.pt`, etc.).
- **Logs**: Training loss and validation AUC curves will be logged to CSV files inside the `./runs` directory.
- **Mixed Precision**: The training loop will automatically detect your NVIDIA RTX GPU and enable Mixed Precision (AMP) using Tensor Cores to speed up the process.

## 4. Evaluation (Post-Training)

After `train_runner.py` finishes executing all 3 seeds, you will evaluate the best model and calibrate the detection threshold using the `04_evaluate.ipynb` notebook.

You can launch Jupyter Notebook to run the evaluation phase:
```bash
jupyter notebook notebooks\04_evaluate.ipynb
```
