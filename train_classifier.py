"""
Classical ML pipeline: extract audio features from m4a files and classify full vs half.
Includes Delta MFCC features, evaluation plots, and STFT spectrogram generation.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
BACKUP_DIR = PROJECT_DIR / "backup_before_temporal"
FEATURES_CSV = PROJECT_DIR / "audio_features.csv"
MODEL_PATH = PROJECT_DIR / "full_half_classifier.joblib"
METRICS_CHART_PATH = FIGURES_DIR / "results_metrics.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "mfcc_feature_importance.png"
SPECTROGRAM_PATH = FIGURES_DIR / "spectrogram_report.png"
PITCH_OVER_TIME_PATH = FIGURES_DIR / "pitch_over_time.png"
REPRESENTATIVE_AUDIO = PROJECT_DIR / "paper_full1.m4a"
PREVIOUS_ACCURACY = 0.84

N_MFCC = 13
SAMPLE_RATE = 22050
MAX_SPECTROGRAM_HZ = 8000
N_FOLDS = 5
N_TEMPORAL_SEGMENTS = 3
SEGMENT_NAMES = ("start", "middle", "end")


def parse_filename(filepath: Path) -> tuple[str, str]:
    """Extract material and label (full/half) from filename."""
    stem = filepath.stem.lower()
    stem = re.sub(r"^[^\w]+", "", stem)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem)

    match = re.search(r"_(full|half)", stem)
    if not match:
        raise ValueError(f"Cannot parse label from filename: {filepath.name}")

    label = match.group(1)
    material = stem[: match.start()].strip("_")
    return material, label


def load_audio(filepath: Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio via ffmpeg (required for m4a on Windows)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-i",
        str(filepath),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    y, loaded_sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
    return y, loaded_sr


def extract_pitch(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate pitch using YIN algorithm; return mean and std of valid frames."""
    f0 = compute_pitch_contour(y, sr)
    valid = f0[np.isfinite(f0) & (f0 > 0)]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.mean(valid)), float(np.std(valid))


def compute_pitch_contour(y: np.ndarray, sr: int) -> np.ndarray:
    """Return frame-wise pitch estimates using YIN."""
    return librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )


def split_into_segments(y: np.ndarray, n_segments: int = N_TEMPORAL_SEGMENTS) -> list[np.ndarray]:
    """Split a waveform into equal-length temporal segments."""
    if y.size == 0:
        return [np.array([], dtype=y.dtype) for _ in range(n_segments)]

    segment_length = max(1, y.size // n_segments)
    segments: list[np.ndarray] = []
    for index in range(n_segments):
        start = index * segment_length
        end = y.size if index == n_segments - 1 else (index + 1) * segment_length
        segments.append(y[start:end])
    return segments


def _add_segment_features(
    features: dict[str, float],
    segment_audio: np.ndarray,
    sr: int,
    segment_name: str,
) -> None:
    """Extract pitch and MFCC statistics for one temporal segment."""
    prefix = segment_name

    if segment_audio.size == 0:
        features[f"{prefix}_pitch_mean"] = 0.0
        features[f"{prefix}_pitch_std"] = 0.0
        for i in range(N_MFCC):
            features[f"{prefix}_mfcc_{i + 1}_mean"] = 0.0
        return

    pitch_mean, pitch_std = extract_pitch(segment_audio, sr)
    segment_mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=N_MFCC)

    features[f"{prefix}_pitch_mean"] = pitch_mean
    features[f"{prefix}_pitch_std"] = pitch_std
    for i in range(N_MFCC):
        features[f"{prefix}_mfcc_{i + 1}_mean"] = float(np.mean(segment_mfccs[i]))


def extract_temporal_segment_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract pitch and MFCC features from start, middle, and end segments."""
    features: dict[str, float] = {}
    segments = split_into_segments(y, N_TEMPORAL_SEGMENTS)
    for segment_name, segment_audio in zip(SEGMENT_NAMES, segments):
        _add_segment_features(features, segment_audio, sr, segment_name)
    return features


def _add_mfcc_features(features: dict[str, float], coeffs: np.ndarray, prefix: str) -> None:
    """Append mean/std statistics for each MFCC (or Delta MFCC) coefficient."""
    for i in range(N_MFCC):
        features[f"{prefix}_{i + 1}_mean"] = float(np.mean(coeffs[i]))
        features[f"{prefix}_{i + 1}_std"] = float(np.std(coeffs[i]))


def extract_features(filepath: Path) -> dict[str, float]:
    """Extract classical audio features including MFCC and Delta MFCC."""
    y, sr = load_audio(filepath, sr=SAMPLE_RATE)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta_mfccs = librosa.feature.delta(mfccs)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms_energy = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    pitch_mean, pitch_std = extract_pitch(y, sr)

    features: dict[str, float] = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "energy_mean": float(np.mean(rms_energy)),
        "energy_std": float(np.std(rms_energy)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }

    _add_mfcc_features(features, mfccs, "mfcc")
    _add_mfcc_features(features, delta_mfccs, "delta_mfcc")
    features.update(extract_temporal_segment_features(y, sr))

    return features


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Scan m4a files, extract features, and build a labeled DataFrame."""
    m4a_files = sorted(data_dir.glob("*.m4a"))
    if not m4a_files:
        raise FileNotFoundError(f"No m4a files found in {data_dir}")

    rows: list[dict] = []
    for filepath in m4a_files:
        material, label = parse_filename(filepath)
        features = extract_features(filepath)
        rows.append(
            {
                "filename": filepath.name,
                "material": material,
                "label": label,
                **features,
            }
        )
        safe_name = filepath.name.encode("ascii", "replace").decode()
        print(f"  Processed: {safe_name} -> material={material}, label={label}")

    return pd.DataFrame(rows)


def build_model(model_type: str = "random_forest") -> Pipeline:
    """Create a scaled classical ML pipeline."""
    if model_type == "svm":
        classifier = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def evaluate_model(X: pd.DataFrame, y: pd.Series, model: Pipeline) -> dict:
    """Run 5-fold stratified cross-validation and compute metrics."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm,
        "labels": labels,
        "y_true": y,
        "y_pred": y_pred,
        "classification_report": classification_report(y, y_pred, zero_division=0),
        "classification_report_dict": report_dict,
    }


def save_metrics_chart(results: dict, output_path: Path) -> None:
    """Save a bar chart of accuracy, precision, and recall."""
    metrics = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor="black", linewidth=0.8)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Performance ({N_FOLDS}-Fold Cross-Validation)")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_heatmap(results: dict, output_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    labels = results["labels"]
    cm = results["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix ({N_FOLDS}-Fold CV)")

    correct = int(np.trace(cm))
    total = int(cm.sum())
    accuracy_pct = results["accuracy"] * 100
    ax.text(
        0.5,
        -0.18,
        f"Accuracy: {accuracy_pct:.1f}% ({correct}/{total} correct)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mfcc_feature_importance(
    model: Pipeline,
    feature_cols: list[str],
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Save a bar chart of the top MFCC-related feature importances."""
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    mfcc_df = importance_df[
        importance_df["feature"].str.contains("mfcc", case=False, regex=False)
    ].sort_values("importance", ascending=False)
    top_features = mfcc_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#3A7D44", edgecolor="black")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} MFCC-Related Feature Importances (Random Forest)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return top_features.sort_values("importance", ascending=False)


def backup_existing_results() -> None:
    """Back up current outputs before regenerating with new features."""
    backup_figures_dir = BACKUP_DIR / "figures"
    backup_figures_dir.mkdir(parents=True, exist_ok=True)

    for source in (FEATURES_CSV, MODEL_PATH, PROJECT_DIR / "report.md"):
        if source.exists():
            shutil.copy2(source, BACKUP_DIR / source.name)

    if FIGURES_DIR.exists():
        for figure_path in FIGURES_DIR.glob("*.png"):
            shutil.copy2(figure_path, backup_figures_dir / figure_path.name)

    metadata = (
        f"Backup created: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Previous accuracy reference: {PREVIOUS_ACCURACY * 100:.1f}%\n"
    )
    (BACKUP_DIR / "backup_notes.txt").write_text(metadata, encoding="utf-8")
    print(f"Existing results backed up to: {BACKUP_DIR}")


def save_pitch_over_time_plot(filepath: Path, output_path: Path) -> None:
    """Plot pitch contour over time with start/middle/end segment boundaries."""
    y, sr = load_audio(filepath, sr=SAMPLE_RATE)
    f0 = compute_pitch_contour(y, sr)
    times = librosa.times_like(f0, sr=sr)
    duration = len(y) / sr
    boundaries = [duration / 3, 2 * duration / 3]

    fig, ax = plt.subplots(figsize=(10, 5))
    valid_mask = np.isfinite(f0) & (f0 > 0)
    ax.plot(times[valid_mask], f0[valid_mask], color="#2E86AB", linewidth=1.5, label="Pitch (YIN)")

    segment_colors = ["#E8F4EA", "#FFF4E6", "#FDECEC"]
    segment_edges = [0.0, boundaries[0], boundaries[1], duration]
    for index, segment_name in enumerate(SEGMENT_NAMES):
        ax.axvspan(
            segment_edges[index],
            segment_edges[index + 1],
            color=segment_colors[index],
            alpha=0.45,
            label=f"{segment_name.capitalize()} segment",
        )
        ax.axvline(boundaries[0], color="gray", linestyle="--", linewidth=1)
        ax.axvline(boundaries[1], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title(f"Pitch Change Over Time\nRepresentative file: {filepath.name}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stft_spectrogram(filepath: Path, output_path: Path) -> None:
    """Perform STFT and save a spectrogram limited to 0-8000 Hz."""
    y, sr = load_audio(filepath, sr=SAMPLE_RATE)

    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    freq_mask = frequencies <= MAX_SPECTROGRAM_HZ

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(
        magnitude_db[freq_mask, :],
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    ax.set_ylim(0, MAX_SPECTROGRAM_HZ)
    ax.set_title(
        f"STFT Spectrogram (0-{MAX_SPECTROGRAM_HZ} Hz)\n"
        f"Representative file: {filepath.name}"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("Audio Classification: Full vs Half (MFCC + Delta + Temporal)")
    print("=" * 60)

    FIGURES_DIR.mkdir(exist_ok=True)
    backup_existing_results()

    print(f"\nScanning m4a files in: {PROJECT_DIR}")
    df = load_dataset(PROJECT_DIR)
    print(f"\nTotal samples: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nMaterial distribution:")
    print(df["material"].value_counts().to_string())

    feature_cols = [col for col in df.columns if col not in ("filename", "material", "label")]
    X = df[feature_cols]
    y = df["label"]
    print(f"\nTotal feature count: {len(feature_cols)}")

    df.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved to: {FEATURES_CSV}")

    model = build_model("random_forest")
    print(f"\nRunning {N_FOLDS}-fold cross-validation (Random Forest)...")
    results = evaluate_model(X, y, model)

    print("\n" + "=" * 60)
    print(f"Evaluation Results ({N_FOLDS}-Fold Cross-Validation)")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy'] * 100:.1f}%)")
    print(f"Precision: {results['precision']:.4f} ({results['precision'] * 100:.1f}%)")
    print(f"Recall:    {results['recall']:.4f} ({results['recall'] * 100:.1f}%)")

    accuracy_delta = results["accuracy"] - PREVIOUS_ACCURACY
    if accuracy_delta > 0:
        comparison = f"improved by {accuracy_delta * 100:.1f} percentage points"
    elif accuracy_delta < 0:
        comparison = f"decreased by {abs(accuracy_delta) * 100:.1f} percentage points"
    else:
        comparison = "unchanged"
    print(
        f"\nAccuracy comparison vs previous baseline ({PREVIOUS_ACCURACY * 100:.1f}%): "
        f"{comparison}."
    )

    print("\nConfusion Matrix:")
    print(f"Labels: {results['labels']}")
    print(results["confusion_matrix"])

    print("\nDetailed Classification Report:")
    print(results["classification_report"])

    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "labels": results["labels"],
            "cv_results": {
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "confusion_matrix": results["confusion_matrix"].tolist(),
                "previous_accuracy": PREVIOUS_ACCURACY,
                "accuracy_delta": accuracy_delta,
            },
        },
        MODEL_PATH,
    )
    print(f"\nTrained model saved to: {MODEL_PATH}")

    print("\nGenerating visualizations...")
    save_metrics_chart(results, METRICS_CHART_PATH)
    print(f"  Metrics chart: {METRICS_CHART_PATH}")

    save_confusion_matrix_heatmap(results, CONFUSION_MATRIX_PATH)
    print(f"  Confusion matrix: {CONFUSION_MATRIX_PATH}")

    top_mfcc = save_mfcc_feature_importance(model, feature_cols, FEATURE_IMPORTANCE_PATH)
    print(f"  MFCC importance chart: {FEATURE_IMPORTANCE_PATH}")
    print("\nTop MFCC-related features:")
    print(top_mfcc.to_string(index=False))

    spectrogram_source = REPRESENTATIVE_AUDIO
    if not spectrogram_source.exists():
        spectrogram_source = next(PROJECT_DIR.glob("*.m4a"))

    print(f"\nGenerating STFT spectrogram from: {spectrogram_source.name}")
    save_stft_spectrogram(spectrogram_source, SPECTROGRAM_PATH)
    print(f"  Spectrogram saved to: {SPECTROGRAM_PATH}")

    print(f"\nGenerating pitch-over-time plot from: {spectrogram_source.name}")
    save_pitch_over_time_plot(spectrogram_source, PITCH_OVER_TIME_PATH)
    print(f"  Pitch plot saved to: {PITCH_OVER_TIME_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()
