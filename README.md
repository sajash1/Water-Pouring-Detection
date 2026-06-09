# Water Pouring Detection Using Classical Machine Learning

## Project Overview

This project presents a classical machine learning solution for detecting the fill state of a water container using acoustic recordings.

The system analyzes the sound produced during water pouring and classifies the container state as either:

- Full
- Half

In addition to classification, the system includes a simple decision layer that converts classification results into practical pouring commands:

- CONTINUE POURING
- STOP POURING

The project was developed as part of a university machine learning and digital signal processing assignment and uses only classical machine learning techniques (no deep learning).

---

## Project Goal

The objective is to investigate whether acoustic characteristics alone can be used to determine the fill level of a container during pouring.

The long-term motivation is an automatic pour-stop system capable of preventing overflow by monitoring sound generated during filling.

---

## Dataset

### Recording Conditions

All recordings were collected under controlled conditions:

- Constant water flow rate
- Fixed microphone distance
- Quiet environment
- Empty container at the beginning of each recording
- Water only

### Dataset Summary

| Property | Value |
|-----------|--------|
| Total Recordings | 50 |
| Classes | Full / Half |
| Samples per Class | 25 / 25 |
| Materials | Glass, Hard Plastic, Paper, Thermos, Thin Plastic |
| Audio Format | .m4a |
| Sampling Rate | 22,050 Hz |

### Directory Structure

```text
recordings/
├── glass_full1.m4a
├── glass_half1.m4a
├── paper_full1.m4a
├── paper_half1.m4a
├── thermos_full1.m4a
└── ...
```

The material type is not used as a model input. The classifier learns to distinguish fill level based only on acoustic properties.

---

## Feature Extraction

The system extracts 105 handcrafted audio features from each recording.

### Global Features

- Pitch Mean
- Pitch Standard Deviation
- Spectral Centroid Mean
- Spectral Centroid Standard Deviation
- RMS Energy Mean
- RMS Energy Standard Deviation
- Zero Crossing Rate Mean
- Zero Crossing Rate Standard Deviation

### MFCC Features

The system computes 13 MFCC coefficients and stores:

- Mean value of each coefficient
- Standard deviation of each coefficient

Total:

- 26 MFCC features

### Delta MFCC Features

First-order derivatives of the MFCC coefficients are extracted to capture temporal spectral changes.

Total:

- 26 Delta MFCC features

### Temporal Segment Features

Each recording is divided into:

- Start
- Middle
- End

For each segment:

- Pitch Mean
- Pitch Standard Deviation
- 13 MFCC Mean Values

Total:

- 45 temporal features

### Total Feature Count

- 105 features per recording

---

## Machine Learning Model

The classification pipeline is:

```text
Audio Recording
        ↓
Feature Extraction
        ↓
StandardScaler
        ↓
Random Forest (200 Trees)
        ↓
Full / Half Prediction
```

### Classifier Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
```

Random Forest was selected because it performs well on small and medium-sized datasets, supports non-linear decision boundaries, and provides interpretable feature importance scores.

---

## Decision Logic

Although the project focuses on classification, the model can also act as a decision-making component for an automatic pouring system.

Decision mapping:

| Classification Output | System Decision |
|----------------------|-----------------|
| Half | CONTINUE POURING |
| Full | STOP POURING |

For future real-time deployment, audio can be analyzed continuously using short overlapping windows.

Example decision rule:

```text
If FULL is detected in 3 consecutive windows:
    STOP POURING
Else:
    CONTINUE POURING
```

This simple strategy reduces the chance of false stop commands caused by noise or occasional misclassifications.

---

## Evaluation

The model is evaluated using:

- 5-Fold Stratified Cross Validation

Performance metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 90.0% |
| Precision | 90.1% |
| Recall | 90.0% |

### Confusion Matrix

| | Predicted Full | Predicted Half |
|---|---|---|
| Actual Full | 23 | 2 |
| Actual Half | 3 | 22 |

Correct predictions:

- 45 / 50

---

## Generated Outputs

Running the training script generates:

```text
audio_features.csv
full_half_classifier.joblib

figures/
├── results_metrics.png
├── confusion_matrix.png
├── mfcc_feature_importance.png
├── spectrogram_report.png
└── pitch_over_time.png
```

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python train_classifier.py
```

The script will:

1. Load recordings from the `recordings/` directory
2. Extract acoustic features
3. Train and evaluate the classifier
4. Generate visualizations
5. Save the trained model

---

## Technologies

- Python
- Librosa
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- FFmpeg

---

## Key Findings

- Fill level significantly affects acoustic resonance.
- MFCC and temporal features provide strong discrimination power.
- Classical machine learning is sufficient for this controlled dataset.
- The system achieves 90% accuracy without deep learning.
- The classifier can serve as the decision engine of a future automatic pour-stop system.

---

## Repository Structure

```text
recordings/
figures/

train_classifier.py
audio_features.csv
full_half_classifier.joblib

report.md
final_report.docx

requirements.txt
README.md
```

---

## Conclusion

This project demonstrates a complete classical machine learning workflow for audio-based water level detection. Using handcrafted acoustic features and a Random Forest classifier, the system successfully distinguishes between full and half-filled containers with 90% accuracy under controlled conditions.

The results suggest that acoustic monitoring is a viable approach for future automated water-pouring systems while remaining computationally simple, interpretable, and easy to deploy.