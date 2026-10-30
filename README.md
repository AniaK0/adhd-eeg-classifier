# ADHD EEG Classifier

A modular machine learning project for EEG signal classification and ADHD detection using a Random Forest classifier and robust subject-independent cross-validation.

## Project Structure

The codebase is organized into clean, independent modules:
- **`loader.py`** – Handles raw data loading, preprocessing, and feature management.
- **`eda.py`** – Performs exploratory data analysis (EDA) and checks class balance.
- **`train.py`** – Trains the Random Forest model using `GroupShuffleSplit` to strictly prevent data leakage across subjects.
- **`main.py`** – The central pipeline coordinating data ingestion, EDA, model training, and evaluation.

## Key Engineering Decisions

1. **Subject-Independent Split (`GroupShuffleSplit`):** 
   To ensure the model learns general EEG signal patterns rather than memorizing specific individuals, data is partitioned based on Subject IDs. This completely eliminates data leakage between training and testing sets.
2. **Performance & Scalability:** 
   Leverages an optimized Random Forest classifier with `n_jobs=-1` to fully utilize multi-core processors when handling large-scale time-series features.
3. **Medical Focus (High Recall):** 
   Optimized to maximize sensitivity (Recall of ~85%) in identifying positive ADHD cases, prioritizing clinical safety and minimizing false negatives over raw accuracy.

## Quick Start

```bash
python main.py