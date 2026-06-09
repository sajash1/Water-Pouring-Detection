# Water Container Audio Classification Using Classical Machine Learning
## A University Project Report: Full vs. Half Fill-Level Detection from Audio

### Project Overview
* **Method:** MFCC, Delta MFCC, temporal segment features, DSP/STFT + Random Forest
* **Dataset:** 50 recordings
* **Features:** 105
* **Evaluation:** 5-fold cross-validation
* **Accuracy:** 90.0% (45/50 correct)

---

## 1. Introduction and Project Goal
This project builds a classical machine learning pipeline to classify audio recordings of water containers as either **full** or **half** based on sound alone. The system uses hand-crafted audio features (MFCC, Delta MFCC, temporal segment features, pitch, spectral centroid, energy, and zero-crossing rate) and a Random Forest classifier.

The practical motivation is automated fill-level monitoring during water pouring. By detecting when a container transitions from half-full to full from its acoustic signature alone, a future system could signal a user or controller to stop pouring before overflow. This report evaluates whether classical audio features are sufficient for that discrimination task on a controlled dataset of 50 recordings.

> **Scope note:** This project uses classical machine learning only — no deep learning or neural networks. Real-time inference is not required; the focus is on a complete, reproducible DSP + ML workflow with honest cross-validation results.

---

## 2. Recording Conditions
All 50 recordings were captured under controlled conditions to reduce variability from environmental and setup factors:
* **Constant flow rate** while pouring water into each container.
* **Fixed microphone distance** from the container for every recording.
* **Quiet environment** with minimal background noise.
* **Empty start:** Each recording started with an empty cup before pouring began.
* **Water only:** No ice, other liquids, or foreign objects in the containers.

These controls help isolate fill-level effects on the acoustic signal. Residual variation still comes from container material (glass, hard plastic, paper, thermos, thin plastic) and minor differences in pour dynamics.

---

## 3. Data Collection
Recordings were stored as `.m4a` files in the project directory. Each filename encodes both the container material and the fill label using the pattern: `{material}_{full|half}{number}.m4a` *(for example, `paper_full1.m4a`)*.

### Dataset Summary
* **Total recordings:** 50
* **Classes:** full (25), half (25) — *perfectly balanced dataset*
* **Materials:** glass, hard_plastic, paper, thermos, thin_plastic
* **Recordings per material:** 10 (5 full + 5 half)
* **Audio format:** .m4a, decoded at 22,050 Hz mono for analysis
* **Feature storage:** Saved to `audio_features.csv` (105 features per sample)

*Note: Materials were not used as model inputs. The classifier must learn fill level from acoustic properties alone, across multiple container types.*

---

## 4. Feature Extraction

### 4.1 MFCC and Delta MFCC
Thirteen MFCC coefficients were computed per recording using `librosa`. For each coefficient, mean and standard deviation across time were stored (26 features). Delta MFCC — the first-order temporal derivative of the MFCC matrix — added 26 more features capturing how spectral shape changes over time.

### 4.2 Global Audio Features
Eight additional global features were extracted: 
* Pitch mean/std (YIN algorithm)
* Spectral centroid mean/std
* RMS energy mean/std
* Zero-crossing rate mean/std

### 4.3 Temporal Segment Features (Start / Middle / End)
Each recording was divided into three equal time segments (start, middle, end). For each segment, pitch mean/std and mean MFCC coefficients (13 values) were computed, yielding 45 temporal features. These capture how pitch and spectral content evolve over the duration of the pour sound — a key cue for fill level.

> **Total Feature Count:** 105 per sample (26 MFCC + 26 Delta MFCC + 8 global + 45 temporal segment). The previous baseline used 60 global features only and achieved 84.0% accuracy.

---

## 5. DSP and STFT Spectrogram Analysis
The Short-Time Fourier Transform (STFT) was applied to a representative recording (`paper_full1.m4a`) with `n_fft=2048` and `hop_length=512`. The spectrogram was visualized in the 0–8000 Hz range to inspect how energy is distributed across time and frequency. Full containers typically show longer, more resonant low-frequency energy compared to half containers.

---

## 6. How the System Makes Decisions
The classification system uses classical machine learning only. No deep learning or neural networks were used, and there is no real-time processing requirement.

### Decision Pipeline
1. **Step 1:** Extract 105 numeric features from each audio file.
2. **Step 2:** Normalize features with `StandardScaler` (zero mean, unit variance).
3. **Step 3:** Pass scaled features to a Random Forest classifier (200 trees).
4. **Step 4:** Each tree votes for full or half; majority vote is the final prediction.

Random Forest handles heterogeneous feature scales well (after scaling), supports non-linear boundaries, and provides interpretable feature importances. Class weights were set to balanced. Temporal segment MFCCs from the middle and end of recordings now rank highest in importance, confirming that fill level affects how frequencies evolve in the later portion of the signal.

### 6.1 Decision Logic for Pour-Stop Applications
While the primary objective of this project is full-versus-half classification, the same classifier can be used as the decision-making component of an automatic water-pouring system.

The system first extracts 105 acoustic features from the recorded audio signal and classifies the recording as either full or half. The classification result is then converted into a practical pouring decision:

If FULL is detected in 3 consecutive windows:
STOP POURING
Else:
CONTINUE POURING

---

## 7. Model Evaluation (5-Fold Cross-Validation)
Model performance was evaluated using 5-fold stratified cross-validation on all 50 recordings. StratifiedKFold preserved the 25/25 class balance in each fold. Predictions from all folds were combined to compute final metrics.

### 7.1 Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 90.0% |
| **Precision (weighted)** | 90.1% |
| **Recall (weighted)** | 90.0% |
| **Correct predictions** | 45 / 50 |

#### Model Comparison

| Model Version | Features | Accuracy |
| :--- | :--- | :--- |
| Baseline | 60 (global only) | 84.0% |
| **Upgraded** | **105 (global + temporal segments)** | **90.0%** |

*Adding start/middle/end pitch and MFCC segment features improved accuracy by 6.0 percentage points compared to the previous 60-feature baseline.*

### 7.2 Confusion Matrix

| | Predicted: full | Predicted: half |
| :--- | :--- | :--- |
| **Actual: full** | 23 | 2 |
| **Actual: half** | 3 | 22 |

*Overall accuracy: 90.0% (45/50 correct). Misclassifications: 2 full samples predicted as half, and 3 half samples predicted as full (5 errors total, down from 8 in the baseline model).*

### 7.3 Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **full** | 0.88 | 0.92 | 0.90 | 25 |
| **half** | 0.92 | 0.88 | 0.90 | 25 |

### 7.4 Top 10 MFCC Feature Importance

| Rank | Feature | Importance (%) |
| :--- | :--- | :--- |
| 1 | middle_mfcc_9_mean | 7.65% |
| 2 | end_mfcc_4_mean | 7.62% |
| 3 | start_mfcc_5_mean | 3.29% |
| 4 | delta_mfcc_3_mean | 3.14% |
| 5 | middle_mfcc_1_mean | 2.77% |
| 6 | delta_mfcc_1_mean | 2.72% |
| 7 | start_mfcc_2_mean | 2.62% |
| 8 | mfcc_1_mean | 2.43% |
| 9 | mfcc_9_mean | 2.21% |
| 10 | mfcc_3_std | 2.17% |

---

## 8. Visualizations
The following figures summarize signal analysis and model performance. All images were generated by `train_classifier.py` and saved in the `figures/` directory.

* **Figure 1:** STFT spectrogram (0–8000 Hz) of representative recording `paper_full1.m4a`.
* **Figure 2:** Pitch change over time for `paper_full1.m4a`. Shaded regions mark start, middle, and end segments used for temporal feature extraction.
* **Figure 3:** Model performance metrics from 5-fold CV. Accuracy: 90.0% (45/50 correct).
* **Figure 4:** Confusion matrix heatmap from 5-fold cross-validation. Accuracy: 90.0% (45/50 correct).
* **Figure 5:** Top 10 MFCC-related feature importances (Random Forest). Temporal segment MFCCs rank highest.

---

## 9. Insights and Conclusions

### 9.1 Key Insights
* **Fill level affects acoustic resonance:** Full and half containers produce detectably different spectral and temporal patterns.
* **Temporal features matter:** Segment MFCC features (middle and end) ranked highest in importance, showing that time-varying frequency content is a strong discriminator.
* **Performance gain:** The upgraded 105-feature model achieved 90.0% accuracy (45/50), up from 84.0% with 60 global features only.
* **Robust generalization:** The model generalized across five materials without using material labels.
* **Classical ML efficiency:** Classical ML (no deep learning) is sufficient for this controlled dataset; real-time deployment is a possible future extension, not a project requirement.

### 9.2 Application to Stopping Water Pouring
Accurate fill-level detection is the first step toward an automated pour-stop system. In a practical deployment, a microphone would monitor the sound of water entering a container; when the classifier predicts the full state with sufficient confidence, it could trigger a valve closure or alert the user to stop pouring. The 90% cross-validation accuracy on this dataset suggests the acoustic approach is viable under controlled conditions, though further data collection and testing across varied environments would be needed before production use.

### 9.3 Conclusions
This project demonstrates a complete classical machine learning workflow for audio-based water level classification. Using 105 hand-crafted features (MFCC, Delta MFCC, and temporal segment features), a Random Forest classifier achieved 90.0% accuracy (45/50 correct) under 5-fold cross-validation with balanced precision and recall. The approach is interpretable, reproducible, and suitable for a university submission in machine learning and digital signal processing.

---

### References and Project Files
* `train_classifier.py` — Main pipeline script
* `audio_features.csv` — Extracted features for 50 recordings (105 features)
* `full_half_classifier.joblib` — Trained model
* `figures/` — Generated PNG visualizations (spectrogram, pitch, metrics, confusion matrix, importance)
* `report.md` — Detailed markdown report
* **Core Libraries:** `librosa`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`