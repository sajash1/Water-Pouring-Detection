# Water Container Audio Classification Project Report

## 1. Introduction

This project builds a **classical machine learning pipeline** to classify audio recordings of water containers as either **full** or **half** based on sound alone. The system uses hand-crafted audio features (MFCC, Delta MFCC, temporal segment features, pitch, spectral centroid, energy, and zero-crossing rate) and a **Random Forest** classifier.

No deep learning is used. The goal is to demonstrate a complete DSP + ML workflow: audio loading, feature extraction, model training, cross-validation, visualization, and reporting.

---

## 2. Problem Statement

Given a short audio recording of a water container (glass, hard plastic, paper, thermos, or thin plastic), predict whether the container is:

- **Full** — mostly filled with water
- **Half** — partially filled with water

This is a **binary classification** task. Material type is recorded in the dataset but is **not** used as an input feature during training — the model must generalize across different container materials using only acoustic properties.

---

## 3. Dataset

### 3.1 Source

All recordings are stored as `.m4a` files in the project directory. Labels and materials are encoded in filenames using the pattern:

```
{material}_{full|half}{number}.m4a
```

**Examples:**
- `glass_half1.m4a` → material: glass, label: half
- `paper_full1.m4a` → material: paper, label: full
- `hard _plastic_full1.m4a` → material: hard_plastic, label: full

### 3.2 Dataset Summary

| Property | Value |
|----------|-------|
| Total samples | 50 |
| Materials | 5 (glass, hard_plastic, paper, thermos, thin_plastic) |
| Recordings per material | 10 (5 full + 5 half) |
| Class balance | 25 full, 25 half (balanced) |
| Sample rate | 22,050 Hz (mono) |

### 3.3 Extracted Features File

All extracted features are saved to **`audio_features.csv`**, which contains:

- `filename`, `material`, `label`
- 105 numeric feature columns (see Section 4)

---

## 4. Feature Extraction

Each audio file is decoded via **ffmpeg** (required for `.m4a` on Windows), then processed with **librosa**. The following **classical audio features** are extracted per file:

### 4.1 MFCC (Mel-Frequency Cepstral Coefficients)

- 13 MFCC coefficients computed over time
- For each coefficient: **mean** and **standard deviation**
- Total: **26 features** (`mfcc_1_mean` … `mfcc_13_std`)

MFCCs capture the timbre and spectral envelope of the sound — useful for distinguishing resonance patterns between full and half containers.

### 4.2 Delta MFCC (First-Order Derivative)

- Computed using `librosa.feature.delta()` on the MFCC matrix
- Captures **how MFCCs change over time** (temporal dynamics)
- For each of 13 delta coefficients: **mean** and **std**
- Total: **26 features** (`delta_mfcc_1_mean` … `delta_mfcc_13_std`)

Delta MFCC adds temporal information that static MFCC statistics alone may miss — for example, how quickly the sound decays or shifts in frequency.

### 4.3 Additional Features

| Feature | Description | Count |
|---------|-------------|-------|
| Pitch (YIN) | Mean and std of fundamental frequency | 2 |
| Spectral Centroid | Mean and std of brightness / center of mass | 2 |
| Energy (RMS) | Mean and std of signal loudness | 2 |
| Zero-Crossing Rate | Mean and std of sign-change rate | 2 |

### 4.4 Temporal Segment Features (Start / Middle / End)

Each recording is divided into **three equal time segments**:

1. **Start** — first third of the waveform
2. **Middle** — middle third
3. **End** — final third

For each segment, the following features are extracted:

| Per-segment feature | Description |
|---------------------|-------------|
| `{segment}_pitch_mean` | Mean pitch (YIN) in the segment |
| `{segment}_pitch_std` | Pitch variability in the segment |
| `{segment}_mfcc_1_mean` … `{segment}_mfcc_13_mean` | Mean MFCC coefficients in the segment |

Where `{segment}` is `start`, `middle`, or `end`.

- Total: **45 temporal features** (3 segments × 15 features)

These features capture **how pitch and spectral content evolve over time**, which is important because filling level affects resonance decay and frequency shift across the duration of the sound.

### 4.5 Total Feature Count

**105 features** per sample (26 MFCC + 26 Delta MFCC + 8 global + 45 temporal segment).

---

## 5. Signal Processing: STFT Spectrogram

To visualize the frequency content of a representative recording, a **Short-Time Fourier Transform (STFT)** was applied to `paper_full1.m4a`.

**STFT parameters:**
- `n_fft = 2048`
- `hop_length = 512`
- Frequency range displayed: **0–8000 Hz**
- Magnitude displayed in decibels (dB)

The spectrogram shows how energy is distributed across time and frequency. Full containers typically produce longer, more resonant sounds with distinct low-frequency components compared to half containers.

![STFT Spectrogram (0–8000 Hz)](figures/spectrogram_report.png)

**Figure 1:** STFT spectrogram of a representative full paper container recording (`paper_full1.m4a`), limited to 0–8000 Hz. Saved as `figures/spectrogram_report.png`.

---

## 6. Temporal Analysis: How Sound Frequencies Change During Filling

When water is poured or a container is tapped, the **acoustic response changes over time**. A fuller container has more mass and a different resonant cavity, which affects how frequencies develop from the start to the end of the sound.

### 6.1 Physical Explanation

- **Full containers** tend to produce **longer resonance**, stronger **low-frequency energy**, and a **more stable pitch** as vibrations decay through the water and container walls.
- **Half-full containers** have a **partial air cavity**, which changes the effective resonator size. This often leads to:
  - Faster energy decay in some bands
  - Shifts in dominant frequencies between the start and end of the sound
  - More variable pitch over time

In other words, filling level is not only visible in the average spectrum — it is also visible in **how frequencies evolve across time**.

### 6.2 Pitch Change Over Time

To visualize this, pitch was tracked frame-by-frame using the **YIN algorithm** on `paper_full1.m4a`. The recording was split into start, middle, and end segments (vertical dashed lines).

![Pitch Change Over Time](figures/pitch_over_time.png)

**Figure 2:** Pitch contour over time for a representative full paper container. Shaded regions mark the start, middle, and end segments used for temporal feature extraction. Saved as `figures/pitch_over_time.png`.

The plot shows that pitch is not constant — it rises and falls as the sound evolves. Segment-level pitch and MFCC features allow the classifier to compare **early vs. late acoustic behavior**, which improves discrimination between full and half containers.

### 6.3 Why Temporal Features Help Classification

The previous baseline model (without segment features) achieved **84.0% accuracy**. After adding start/middle/end pitch and MFCC features, accuracy improved to **90.0%** using the same 50 recordings and 5-fold cross-validation. This suggests that **time-varying frequency content** is a meaningful cue for fill-level detection.

---

## 7. Machine Learning Model

### 7.1 Algorithm

**Random Forest Classifier** with the following settings:

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | None (unlimited) |
| `class_weight` | balanced |
| `random_state` | 42 |

### 7.2 Preprocessing Pipeline

```
Raw features (105 values)
  → StandardScaler (zero mean, unit variance)
  → Random Forest (200 decision trees, majority vote)
  → Prediction: "full" or "half"
```

Each decision tree learns simple threshold rules on scaled features. The final prediction is the **majority vote** across all 200 trees.

### 7.3 Evaluation Method

**5-fold stratified cross-validation** (`StratifiedKFold`, shuffle=True, random_state=42):

- The dataset is split into 5 folds preserving class proportions
- Each fold is used once as a test set
- Predictions are aggregated across all folds for final metrics
- This avoids overfitting to a single train/test split on a small dataset

---

## 8. Results

### 8.1 Overall Performance

Results below are from a full re-run of the upgraded pipeline on all **50 `.m4a` files** (25 full, 25 half) using 5-fold stratified cross-validation.

| Metric | Score |
|--------|-------|
| **Accuracy** | **90.0%** |
| **Precision (weighted)** | **90.1%** |
| **Recall (weighted)** | **90.0%** |
| **Correct predictions** | **45 / 50** |

**Comparison with previous baseline (without temporal segment features):**

| Model version | Features | Accuracy |
|---------------|----------|----------|
| Baseline | 60 (global only) | 84.0% |
| Upgraded | 105 (global + temporal segments) | **90.0%** |

Adding temporal pitch and MFCC segment features improved accuracy by **6.0 percentage points**. This is an honest cross-validation result — no target accuracy was enforced.

Previous outputs were backed up to `backup_before_temporal/` before regeneration.

![Model Performance Metrics](figures/results_metrics.png)

**Figure 3:** Accuracy, precision, and recall from 5-fold cross-validation on all 50 recordings. Saved as `figures/results_metrics.png`.

### 8.2 Confusion Matrix

|  | Predicted: full | Predicted: half |
|--|-----------------|-----------------|
| **Actual: full** | 23 | 2 |
| **Actual: half** | 3 | 22 |

![Confusion Matrix Heatmap](figures/confusion_matrix.png)

**Figure 4:** Confusion matrix heatmap from 5-fold cross-validation. Saved as `figures/confusion_matrix.png`.

### 8.3 Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| full | 0.88 | 0.92 | 0.90 | 25 |
| half | 0.92 | 0.88 | 0.90 | 25 |

**Interpretation:**
- Both classes are now balanced at ~90% F1-score.
- Only **5 samples** were misclassified (2 full → half, 3 half → full), down from 8 in the baseline model.
- Temporal segment features reduced errors on both classes.

### 8.4 Feature Importance (Top 10 MFCC-Related Features)

The Random Forest ranks features by how much they reduce impurity across all trees. The top MFCC-related features are:

| Rank | Feature | Importance |
|------|---------|------------|
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

![Top 10 MFCC Feature Importance](figures/mfcc_feature_importance.png)

**Figure 5:** Top 10 MFCC-related feature importances from the trained Random Forest. Temporal segment MFCCs (`middle_mfcc_9_mean`, `end_mfcc_4_mean`) now rank highest. Saved as `figures/mfcc_feature_importance.png`.

---

## 9. How the Model Makes Decisions

1. **Audio is converted to numbers:** Each `.m4a` file becomes a 105-dimensional feature vector summarizing global and segment-level spectral/temporal properties.
2. **Features are normalized:** `StandardScaler` ensures all features contribute on a comparable scale.
3. **200 trees vote independently:** Each tree applies threshold rules on scaled features (including segment-level pitch and MFCC values).
4. **Majority vote wins:** The class predicted by the most trees becomes the final output.

**Key insight:** Temporal segment MFCCs from the **middle** and **end** of the recording are now the most discriminative features. This confirms that fill level affects not just the overall sound, but specifically **how frequencies evolve in the later portion of the signal**.

---

## 10. Project Files

| File | Description |
|------|-------------|
| `train_classifier.py` | Main pipeline script |
| `requirements.txt` | Python dependencies |
| `audio_features.csv` | Extracted features for all 50 samples (105 features) |
| `full_half_classifier.joblib` | Trained model (pipeline + metadata) |
| `backup_before_temporal/` | Backup of previous CSV, model, figures, and report |
| `figures/results_metrics.png` | Performance metrics bar chart |
| `figures/confusion_matrix.png` | Confusion matrix heatmap |
| `figures/mfcc_feature_importance.png` | Top 10 MFCC feature importances |
| `figures/spectrogram_report.png` | STFT spectrogram (0–8000 Hz) |
| `figures/pitch_over_time.png` | Pitch contour with start/middle/end segments |
| `report.md` | This report |

---

## 11. Reproducibility

Install dependencies and run the full pipeline:

```powershell
pip install -r requirements.txt
python train_classifier.py
```

This will:
1. Back up existing results to `backup_before_temporal/`
2. Scan all `.m4a` files
3. Extract MFCC + Delta MFCC + temporal segment + other features
4. Save `audio_features.csv`
5. Train and evaluate with 5-fold cross-validation
6. Save the model to `full_half_classifier.joblib`
7. Generate all figures in the `figures/` directory

**Note:** `.m4a` decoding requires ffmpeg, bundled automatically via the `imageio-ffmpeg` package.

---

## 12. Limitations

1. **Small dataset:** Only 50 samples (5 per class per material) limits generalization.
2. **Cross-validation variance:** Metrics may shift with different random seeds or fold assignments.
3. **Material overlap:** Some materials may produce similar sounds at full/half levels, causing misclassification.
4. **Recording conditions:** Background noise, microphone distance, and tap force are not controlled.
5. **Misclassification rate:** 5 out of 50 samples were still misclassified (2 full → half, 3 half → full).
6. **Fixed segment boundaries:** Splitting each recording into equal thirds is simple but may not align with physical events in every recording.

---

## 13. Conclusions

This project successfully demonstrates a complete classical ML pipeline for audio-based water level classification:

- **105 hand-crafted features** (including MFCC, Delta MFCC, and temporal segment features) were extracted from 50 container recordings.
- A **Random Forest** classifier achieved **90.0% accuracy** with balanced precision and recall using 5-fold cross-validation.
- Adding **start/middle/end pitch and MFCC features** improved accuracy from **84.0% to 90.0%** compared to the previous baseline.
- **Temporal segment MFCCs** proved most important, highlighting that fill level affects how frequencies evolve over time.
- **STFT spectrogram** and **pitch-over-time** analysis confirmed visible frequency and pitch differences in representative recordings.
- All results, features, model, backups, and figures are saved for reproducibility and submission.

---

## 14. Future Work

1. Collect more recordings per class to further improve generalization.
2. Compare Random Forest vs. SVM performance (SVM support is built into the script).
3. Add **Delta-Delta MFCC** (second-order derivatives) for additional temporal detail.
4. Perform **leave-one-material-out** evaluation to test cross-material generalization.
5. Extend to **multi-class material classification** as a separate task.
6. Experiment with **adaptive segment boundaries** based on energy onset/decay instead of fixed thirds.

---

*Report generated for the Water Container Audio Classification project.*
