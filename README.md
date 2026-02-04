# Audio Event Classification using Deep CNN

Deep Convolutional Neural Network for classifying labeled audio events from web audio data. This project implements a VGG-style CNN architecture with MFCC preprocessing and LSTM integration for robust sound event detection and classification.

## Project Overview

This project addresses the challenge of audio event classification using deep learning techniques. By leveraging the success of CNNs in image processing, we apply similar architectures to audio spectrograms for sound classification. The model achieves **82.5% accuracy** on 525 audio event classes.

**Course**: EECE 7398 - Advances in Deep Learning  
**Institution**: Northeastern University  
**Author**: Dharm Mehta

##  Key Features

- **VGG-style CNN Architecture**: 6-block deep convolutional network
- **MFCC Preprocessing**: Mel-Frequency Cepstral Coefficients for audio feature extraction
- **LSTM Integration**: Long Short-Term Memory for temporal dependency modeling
- **Multi-label Classification**: Binary cross-entropy and focal loss for handling multiple labels
- **Layer-wise Relevance Propagation (LRP)**: Explainability through propagation analysis
- **Strong Label Assumption Training (SLAT)**: Efficient training on segmented audio
- **Adam Optimization**: Fast convergence with adaptive learning rates

##  Architecture

### Network Structure
```
Input (Log-Mel Spectrogram: 896×128)
    ↓
Block 1: Conv(16) → Conv(16) → MaxPool → [32×224×64]
    ↓
Block 2: Conv(32) → Conv(32) → MaxPool → [64×112×32]
    ↓
Block 3: Conv(64) → Conv(64) → MaxPool → [128×56×16]
    ↓
Block 4: Conv(128) → Conv(128) → MaxPool → [256×28×8]
    ↓
Block 5: Conv(256) → Conv(256) → MaxPool → [512×14×4]
    ↓
Block 6: Conv(512) → MaxPool → [512×7×2]
    ↓
F1: Conv(1024) 2×2 filters → [1024×1×1]
    ↓
F2: Conv(525) 1×1 filters → Sigmoid → [525×K×1]
    ↓
Global Pooling → Output [525×1]
```

### Architecture Details

| Layer | Filters | Kernel | Stride | Padding | Output |
|-------|---------|--------|--------|---------|--------|
| **Block 1** | 16 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **Block 2** | 32 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **Block 3** | 64 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **Block 4** | 128 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **Block 5** | 256 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **Block 6** | 512 | 3×3 | 1 | 1 | BatchNorm + ReLU |
| **F1** | 1024 | 2×2 | 1 | 0 | ReLU |
| **F2** | 525 | 1×1 | 1 | 0 | Sigmoid |

**Key Features:**
- All convolutional layers use 3×3 filters (except F1: 2×2, F2: 1×1)
- Max pooling: 2×2 window, stride 2
- Batch normalization after each convolution
- ReLU activation throughout
- Fully convolutional design for variable-length audio



## 💻 Usage

### 1. Audio Preprocessing

The preprocessing pipeline extracts MFCC features:
```python
import librosa

# Load audio
y, sr = librosa.load('audio_file.wav', sr=44100)

# Extract Mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_fft=1024,
    hop_length=512, 
    n_mels=128
)

# Convert to log scale
log_mel = librosa.power_to_db(mel_spec)
```

### 2. Training

Train the model with default configuration:
```python
from project_final import Config, get_1d_conv_model

# Configure
config = Config(
    sampling_rate=16000,
    audio_duration=2,
    n_classes=41,
    n_folds=10,
    learning_rate=0.001,
    max_epochs=50
)

# Create model
model = get_1d_conv_model(config)

# Train
history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=config.max_epochs,
    callbacks=[checkpoint, early_stopping, tensorboard]
)
```

### 3. Inference
```python
# Load trained model
model.load_weights('best_model.h5')

# Predict on new audio
predictions = model.predict(test_audio)
```

### 4. Feature Extraction

Extract features from pretrained model:
```python
from project_final import main

# Extract features
features = main('audio_file.wav', srate=44100)
print(f"Feature shape: {features.shape}")
```

##  Model Components

### MFCC Feature Extraction

**Steps:**
1. **Pre-emphasis**: Amplify high-frequency components
2. **Framing**: Divide signal into 22ms windows with 10.5ms overlap
3. **Windowing**: Apply Hamming window
4. **FFT**: Fast Fourier Transform (n_fft=1024)
5. **Mel Filter Bank**: 127 Mel bands
6. **Logarithm**: Convert to dB scale
7. **DCT**: Discrete Cosine Transform

### LSTM Integration
```python
# LSTM unit update equations
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Cell candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t ⊙ tanh(c_t)  # Hidden state
```

##  Training Configuration

### Hyperparameters
```python
# Model Configuration
sampling_rate = 16000  # Hz
audio_duration = 2     # seconds
n_folds = 10          # K-fold cross-validation
learning_rate = 0.001 # Adam optimizer
max_epochs = 50
batch_size = 64

# Audio Processing
n_fft = 1024
hop_length = 512
n_mels = 128
segment_size = 128    # frames
segment_overlap = 64  # frames
```

### Loss Functions

**Binary Cross-Entropy Loss:**
```python
CE = -Σ[t_i·log(s_i) + (1-t_i)·log(1-s_i)]
```

**Focal Loss** (for class imbalance):
```python
FL = -Σ(1-s_i)^γ · t_i · log(s_i)
```
Where γ=2 by default to down-weight easy examples.

### Optimization

**Adam Optimizer:**
- Combines benefits of AdaGrad + RMSprop
- Adaptive learning rates per parameter
- Efficient on sparse gradients
- Less memory required
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8

##  Performance Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 82.5% |
| **Mean AUC (MAUC)** | 0.600 |
| **Mean AP (MAP)** | 0.651 |
| **Training Speed** | 39% faster than baseline |
| **Inference Speed** | 33% faster than baseline |

### Best Performing Classes

| Event | AP | AUC |
|-------|-----|-----|
| Music | 0.728 | 0.749 |
| Voice (Civil Defense) | 0.671 | 0.641 |
| Purr (Cat) | 0.575 | 0.600 |
| Battle Cry | 0.575 | 0.651 |

### Challenging Classes

| Event | AP | AUC |
|-------|-----|-----|
| Scrape | 0.0058 | 0.0092 |
| Gurgling | 0.0111 | 0.0125 |
| Noise | 0.0116 | 0.0107 |

### Improvements from SLAT

- **MAUC**: +1.2 absolute (1.3% relative)
- **MAP**: +4.6 absolute (27.5% relative)
- **Low-performance classes**: Doubled AP (0.0097 → 0.0203)
- **High-performance classes**: 8.5% relative improvement

##  Explainability: Layer-wise Relevance Propagation

### LRP Algorithm

Propagates relevance scores from output back to input:
```python
# Relevance conservation
Σ_i R_i = ... = Σ_j R_j^(l+1) = ... = f(x)

# Redistribution rule
R_j = Σ_k (α_jw_jk / Σ_a,j α_jw_jk) · R_k
```

**Benefits:**
- Identifies which input features contribute to predictions
- Visualizes important time-frequency regions
- Helps debug model decisions
- Improves interpretability



##  Key Innovations

### 1. Strong Label Assumption Training (SLAT)

- Divides audio into overlapping segments
- Assigns same label to all segments
- Enables temporal localization
- Faster training and inference

### 2. Fully Convolutional Design

- No dense layers (except optional)
- Handles variable-length audio
- Enables transfer learning
- Segment-level outputs

### 3. Multi-label Classification

- Binary cross-entropy per class
- Independent class predictions
- Handles co-occurring sounds
- Focal loss for class imbalance

##  Known Limitations

1. **Fixed-length training**: Model trained on 2-second segments
2. **Variable-length inference**: Lower accuracy on varying durations
3. **Class imbalance**: Some classes have few examples
4. **Computational cost**: Deep architecture requires GPU
5. **Transfer learning**: Limited success on small datasets


## 📚 Technical Details

### Segment Processing

- **Input**: 896 log-mel frames (128 Mel bands)
- **Segment size**: 128 frames (~1.5 seconds)
- **Hop size**: 64 frames (~0.75 seconds)
- **Output**: K=13 segment-level predictions
- **Aggregation**: Global max/average pooling

### Data Augmentation

- Time shifting
- Pitch shifting
- Speed perturbation
- Background noise addition
- SpecAugment

##  Evaluation Metrics

### Classification Metrics

- **AUC (Area Under ROC Curve)**: Overall discrimination ability
- **AP (Average Precision)**: Precision-recall trade-off
- **Accuracy**: Correct predictions / Total predictions

### Temporal Metrics

- **Segment-level accuracy**: Frame-wise predictions
- **Event detection**: Onset/offset timing
- **F1-score**: Harmonic mean of precision and recall






---

**Note**: This is an academic research project demonstrating deep learning for audio classification. The code includes integrations with Google Drive for Colab environments.
