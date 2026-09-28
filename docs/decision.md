# Project Decision Log

This document logs all major technical and architectural decisions made throughout the project's lifecycle.

---

### [DECISION-001] Transition to Local Training Environment
- **Date:** 2026-09-29
- **Status:** IMPLEMENTED
- **Category:** ARCHITECTURE
- **Files Changed:** `README.md`, `SYSTEM_ARCHITECTURE.md`, `docs/implementation.md`, `summary.md`
- **Reason:** The original plan relied on Google Colab and Google Drive for training and storage. Training the dataset locally is preferred for better control, stability, and unrestricted execution time.
- **Consideration:** Considered keeping the Colab workflow as a secondary option, but maintaining dual setups creates unnecessary overhead. Considered migrating to an enterprise cloud solution (AWS/GCP), but local execution is more straightforward for immediate capstone needs.
- **Decision:** Decided to fully strip out Google Colab and Google Drive references and rely entirely on local Python execution with local disk storage (`data/`, `checkpoints/`, `runs/`, `outputs/`).
- **Code Change Summary:** Updated all project markdown documentation to reflect local paths and local environment setups instead of Colab notebooks.

---

### [DECISION-002] Create NVIDIA GPU Environment Setup
- **Date:** 2026-09-29
- **Status:** IMPLEMENTED
- **Category:** CONFIG
- **Files Changed:** `requirements.txt`, `setup_env.py`
- **Reason:** With the move to local training, a reproducible environment setup is required specifically tailored for NVIDIA GPUs (CUDA), as 3D CNN training is computationally heavy.
- **Consideration:** Considered using Conda `environment.yml` but decided on a simpler `requirements.txt` paired with a Python setup script to avoid enforcing a specific package manager.
- **Decision:** Created a `requirements.txt` specifically pinning PyTorch with CUDA 12.1 (`cu121`), and a `setup_env.py` script to generate necessary project directories and verify CUDA availability.
- **Code Change Summary:** Created `requirements.txt` with PyTorch CUDA wheels and `setup_env.py` for directory scaffolding and environment logging.
