# Water Pouring Detection and Classification 
 
An advanced academic engineering project utilizing audio signal processing and machine learning models to automatically detect and classify water pouring events. 
 
## Project Highlights 
* **Model Accuracy:** Achieved **90%% validation accuracy** using optimized feature extraction. 
* **Data Processing:** Features extracted from audio recordings are structured in audio_features.csv. 
 
## Repository Structure 
* train_classifier.py - Core machine learning script. 
* create_final_report.py - Script responsible for compiling results. 
* audio_features.csv - The extracted audio feature table. 
* final_report.docx - Full comprehensive academic report. 
## ?? Methodology and Signal Processing 
The project pipeline processes raw audio files through several advanced engineering steps: 
1. **Audio Preprocessing:** Short-Time Fourier Transform (STFT) and noise reduction. 
2. **Feature Extraction:** Extracting key audio descriptors including Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid, and Zero Crossing Rate. 
3. **Model Training:** Training a Random Forest / SVM classifier to identify the unique acoustic signature of pouring water. 
 
## ?? Experimental Results 
* **Dataset Size:** Built from dozens of unique real-world audio recordings. 
* **Classification Performance:** The model successfully distinguished between pouring events and background noise with a **90%% success rate**. 
* **Confusion Matrix:** High precision and recall rates across all tested audio segments. 
 
## ?? How To Run The Project 
1. Install dependencies: `pip install -r requirements.txt` 
2. Run feature extraction and training: `python train_classifier.py` 
3. Generate the final submission report: `python create_final_report.py` 
 
## ?? Academic Credits 
* **Author:** Saja Sharkia 
* **Project Status:** Completed - Final Academic Submission (2026). 
