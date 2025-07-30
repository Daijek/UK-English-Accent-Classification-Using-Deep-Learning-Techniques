# 🚦 English Accent Identification Across Six UK Regions

_By Ijezie Daniel Ekenedilichukwu_

---

📣 **Abstract**  
Accents are rich in phonetic detail and help distinguish speakers from different regions. We compare nine deep-learning models across three feature sets—MFCC, YamNet embeddings, and VGGish embeddings—to classify six British accents (Irish, Southern, Northern, Midlands, Scottish, Welsh). Our best performer is a CNN trained on MFCCs (95.10% accuracy, 94.18% F1), followed by a VGGish-LSTM (86.65% accuracy, 86.38% F1). We also demonstrate excellent transfer-learning performance on a held-out LibriTTS-British accents dataset (95.87% accuracy).

---

📖 **Table of Contents**  
1. [🚀 Motivation](#-motivation)  
2. [🗂️ Project Structure](#️-project-structure)  
3. [🛠️ Setup & Installation](#️-setup--installation)  
4. [📊 Data Pipeline](#-data-pipeline)  
   - [Data Preparation & Augmentation](#data-preparation--augmentation)  
5. [🔬 Feature Extraction & Modeling](#-feature-extraction--modeling)  
   - [MFCC Features & Models](#mfcc-features--models)  
   - [YamNet Embeddings & Models](#yamnet-embeddings--models)  
   - [VGGish Embeddings & Models](#vggish-embeddings--models)  
6. [🏆 Results Summary](#-results-summary)  
7. [💬 Discussion & Literature Comparison](#-discussion--literature-comparison)  
8. [🔮 Future Work](#-future-work)  
9. [🎉 Usage Examples](#-usage-examples)  
10. [📜 License & Acknowledgments](#-license--acknowledgments)

---

## 🚀 Motivation

> “Accents carry social, cultural, and phonetic information—building a robust accent identification system can power language learning, forensic phonetics, and inclusive speech technologies.”  

Despite advances in audio classification, the comparative performance of pretrained audio-feature extractors (YamNet, VGGish) vs. classic MFCC pipelines for accent recognition is under-explored. This project fills that gap and proposes a state-of-the-art CNN on MFCC features, along with strong transfer-learning results.

---

## 🗂️ Project Structure

├── 00_Data_Preprocessing_Augmentation.ipynb
├── 01_YamNet_Embeddings_and_Models.ipynb
├── 02_VGGish_Embeddings_and_Models.ipynb
├── 03_MFCC_Features_and_Models.ipynb
├── data/
│ ├── raw/
│ ├── train/
│ ├── test/
│ └── augmented/
├── notebooks/
├── models/
├── np_save/
│ ├── yamnet_embeddings/
│ └── vggish_embeddings/
├── Pickle Objects/
└── README.md

## 🛠️ Setup & Installation

1. Clone the repo:  
   ```
   git clone https://github.com/yourusername/accent-identification-uk.git
   cd accent-identification-uk
   ```
2. Create a virtual environment & install dependencies:
   ```
   python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
   ```
3. Dependencies include:
   - numpy, pandas, scikit-learn
   - librosa, soundfile, audiomentations
   - tensorflow, tensorflow-hub, keras
   - matplotlib, seaborn, folium
   - mlxtend, pydub, tqdm
  
## 📊 Data Pipeline
![data popeline](https://github.com/Daijek/UK-English-Accent-Classification-Using-Deep-Learning-Techniques/blob/main/images/data%20pipeline.png?raw=true)

### Data Preparation & Augmentation
**Notebook:** `00_Data_Preprocessing_Augmentation.ipynb`

### Datasets
- **Crowdsourced UK & Ireland English Dialect**  
  17,877 recordings across 6 accents (male + female merged)
- **LibriTTS-British Accents**  
  581 recordings (Irish, Scottish, Welsh) for transfer-learning evaluation

### Preprocessing Steps
1. Merge male/female classes for each accent
2. Flatten nested LibriTTS directories into four top-level accent folders
3. 80/20 train-test stratified split via `move_20_percent_to_test()`
4. **Augmentation** (train only) using `audiomentations`:
   - Gaussian noise (`p=0.2`) **OR** real background noise (`p=0.8`)
   - Pitch shift ±8 semitones (+ 40% noise)
   - High-pass filter (2–4 kHz)
```
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, PitchShift, HighPassFilter
```

## 🔬 Feature Extraction & Modeling
### MFCC Features & Models
**Notebook:** `03_MFCC_Features_and_Models.ipynb`

#### Extraction
- Resample to 22,050 Hz
- Extract 40 MFCC coefficients with:
  - `hop_length=512`
  - Normalization via `extract_mfcc_features` class
- Visualize waveforms & spectrograms
![MFCC](https://github.com/Daijek/UK-English-Accent-Classification-Using-Deep-Learning-Techniques/blob/main/images/MFCC%20spectrogram.png?raw=true)

#### Data Prep
1. Map accent labels → integers (0–5)
2. Combine `train` + `augmented_train` sets
3. Stratified 80/20 train/validation split
4. Pad sequences to max length (868 frames)
5. Save processed data in `Pickle Objects/prepared_data`

#### Models
| Model Type                  | Architecture                                     | Performance Metrics                     |
|-----------------------------|--------------------------------------------------|-----------------------------------------|
| **CNN**                     | 4×Conv2D → Pooling → Dropout → Flatten → Dense   | Test Acc: 95.10%<br>F1: 94.18%<br>Recall: 96.78%<br>Precision: 92.05% |
| **LSTM**                    | Masking → Stacked LSTM → Dense                   | Accuracy: 80.18%                        |
| **MLP** (Multi-Layer Perceptron) | Dense → Dropout → Dense              | Accuracy: 69.77%                        |

#### Transfer Learning
- Fine-tuned MFCC-CNN on LibriTTS accents (3 classes):
  - **Accuracy:** 95.87%
  - **F1 Score:** 96.16%
```
# Sample MFCC CNN architecture
model = Sequential([
  Conv2D(32,3,padding='same',activation='relu',input_shape=(868,40,1)),
  MaxPooling2D(), Dropout(0.5),
  Conv2D(64,3,activation='relu'), MaxPooling2D(), Dropout(0.5),
  Flatten(), Dense(256,'relu'), Dropout(0.5), Dense(6,'softmax')
])
```
# 🔬 YamNet Embeddings & Models
**Notebook:** `01_YamNet_Embeddings_and_Models.ipynb`

### Extraction
- Use TensorFlow-Hub YamNet model → 1024-dim frame embeddings
- Resample to 16,000 Hz + normalize
- Extract embeddings for `train`, `test`, `augmented_train` → pickle DataFrames

### Data Prep
1. Map accent labels → integers (0–5)
2. Pad sequences to 41 frames × 1024 dimensions
3. Save as `.npy` files

### Models
| Model | Accuracy |
|-------|----------|
| CNN   | 52.11%   |
| LSTM  | 79.35%   |
| MLP   | 58.40%   |

### YamNet Flow
![Yamnet](https://github.com/Daijek/UK-English-Accent-Classification-Using-Deep-Learning-Techniques/blob/main/images/Yamnet%20flow.png?raw=true)

---

# 🔬 VGGish Embeddings & Models
**Notebook:** `02_VGGish_Embeddings_and_Models.ipynb`

### Extraction
- TensorFlow-Hub VGGish model → 128-dim frame embeddings
- Same resampling & normalization as YamNet

### Data Prep
1. Map accent labels → integers (0–5)
2. Pad to max 20 frames × 128 dimensions
3. Save as `.npy` files

### Models
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| CNN   | 69.91%   | -        |
| LSTM  | 86.65%   | 86.38%   |
| MLP   | 82.06%   | -        |

### VGGish Flow
![vggish](https://github.com/Daijek/UK-English-Accent-Classification-Using-Deep-Learning-Techniques/blob/main/images/vggish%20flow.png?raw=true)

---

# 🏆 Results Summary
| Feature Set | Model | Accuracy | Precision | Recall | F1-Score |
|-------------|-------|----------|-----------|--------|----------|
| MFCC        | CNN   | 95.10%   | 92.05%    | 96.78% | 94.18%   |
| MFCC        | LSTM  | 80.18%   | 79.34%    | 88.14% | 82.33%   |
| MFCC        | MLP   | 69.77%   | 74.09%    | 85.94% | 79.58%   |
| YamNet      | CNN   | 52.11%   | 49.85%    | 63.25% | 50.73%   |
| YamNet      | LSTM  | 79.35%   | 76.07%    | 78.37% | 76.91%   |
| YamNet      | MLP   | 58.40%   | 55.82%    | 71.35% | 59.29%   |
| VGGish      | CNN   | 69.91%   | 66.67%    | 77.44% | 69.78%   |
| VGGish      | LSTM  | 86.65%   | 85.70%    | 87.70% | 86.38%   |
| VGGish      | MLP   | 82.06%   | 78.01%    | 86.12% | 81.14%   |

---

# 💬 Discussion & Literature Comparison
- Our MFCC-CNN outperforms prior studies (Cetin 2022: ~93% Acc)
- VGGish-LSTM shows strong transfer-learning capacity
- YamNet embeddings underperform vs. MFCC & VGGish for accent tasks
- Class imbalance handled via weights → high per-class F1 (Midlands: 87.62% F1)

# 🔮 Future Work
1. Train VGGish variant on MFCC inputs
2. Robustness to noisy speech via advanced augmentation/denoising
3. Extend to other languages' accent classification
4. Real-time inference API & interactive dashboards

---

## 🎉 Usage Examples
### Single-File Accent Prediction (MFCC-CNN)
```
from pydub import AudioSegment
from mfcc_model import load_mfcc_cnn, predict_accent

model = load_mfcc_cnn("models/mfcc_cnn.pkl")
result = predict_accent("/path/to/sample.wav", model, feature="MFCC_CNN")
print(result["Prediction"], result["confidence_levels"])
# result["play_audio"].play()
```

## 📜 License & Acknowledgments

**License:** [MIT License](https://opensource.org/licenses/MIT)

### Data Sources
- [Crowdsourced UK & Ireland English Dialect](https://example.com) (Open Government Licence v1.0)  
- [LibriTTS-British Accents](https://example.com) (CC BY-NC-SA 4.0)

### Acknowledgments
Special thanks to:
- The TensorFlow & TF-Hub teams for YamNet/VGGish models
- Audiomentations library authors for audio augmentation tools
- University of Hull for providing environmental noise samples
- The wider open-source community for invaluable contributions

> "Accents are windows into culture—this pipeline illuminates them with modern deep learning." 🎤🌍
   
