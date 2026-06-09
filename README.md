## ?? Audio Classification Classes 
The acoustic models are trained to classify water pouring into 5 distinct target classes based on audio recording signatures: 
* `paper_pouring` - Audio events recorded using the paper cup. 
* `glass_pouring` - Audio events recorded using the faceted glass tumbler. 
* `thermos_pouring` - Audio events recorded using the insulated green thermos. 
* `large_plastic_pouring` - Audio events recorded using the large clear plastic cup. 
* `small_plastic_pouring` - Audio events recorded using the thin small plastic cup. 
 
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
 
## ?? Updated Vessel Classification 
The dataset successfully includes the updated acoustic categories: 
* **Hard Plastic Cup** - Formed plastic cup with complex resonant modes. 
* **Thin Plastic Cup** - Lightweight thin-walled plastic cup with high-frequency modes. 
 
