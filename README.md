# Time Series Anomaly Detection with Deep Learning

A comprehensive comparative analysis of three deep learning architectures for multivariate time series anomaly detection in spacecraft telemetry data.

## 🎯 Overview

This project implements and compares three state-of-the-art deep learning models for unsupervised anomaly detection:
- **LSTM-VAE**: LSTM Variational Autoencoder with regularized latent space
- **LSTM-Attention**: Sequence-to-sequence LSTM Autoencoder with Bahdanau attention mechanism  
- **Anomaly Transformer**: Transformer architecture with learnable Gaussian-prior attention

The models are evaluated on NASA's SMAP-MSL spacecraft telemetry dataset, providing practical insights for industrial anomaly detection applications.

## 📊 Key Results

| Model | Precision | Recall | F1-Score | AUROC | AUPRC | Training Time |
|-------|-----------|--------|----------|--------|-------|---------------|
| **LSTM-VAE** | **44.7%** | 45.6% | **42.6%** | **67.8%** | **45.4%** | ~2 hours |
| LSTM-Attention | 42.6% | 45.0% | 41.2% | 62.3% | 43.9% | ~2 hours |
| Anomaly Transformer | 33.8% | **49.6%** | 37.7% | - | - | ~6 hours |

**Key Findings:**
- **LSTM-VAE** offers the best precision-recall balance and probabilistic metrics
- **Anomaly Transformer** achieves highest recall but with more false positives
- **LSTM-Attention** provides good performance with smallest model size
- All models excel on D-series channels (F1 > 80%) and struggle with E-series channels

## 🗂️ Dataset

**NASA SMAP-MSL Spacecraft Telemetry Dataset**
- **Sources**: Soil Moisture Active Passive (SMAP) satellite + Mars Science Laboratory (MSL) rover
- **Channels**: 82 total (55 SMAP + 27 MSL), 28 used for evaluation (≥100 anomaly windows)
- **Features**: 25 anonymized sensor variables per channel
- **Temporal**: 1-minute sampling, 2,880 timesteps (2 days)
- **Anomaly Types**: Point anomalies and contextual anomalies
- **Format**: Pre-normalized .npy files

### Data Structure
```
data/
├── train/          # Training data (.npy files)
├── test/           # Test data (.npy files)
├── 2018-05-19_15.00.10/  # Model outputs directory
└── labeled_anomalies.csv  # Ground truth labels
```

## 🏗️ Model Architectures

### 1. LSTM-VAE
- **Encoder**: Single-layer LSTM (128 hidden units) → latent space (μ, σ²)
- **Latent**: 32-dimensional diagonal Gaussian distribution
- **Decoder**: Symmetric LSTM architecture with reparameterization trick
- **Loss**: MSE reconstruction + β-weighted KL divergence (β=0.1)

### 2. LSTM-Attention  
- **Encoder**: Two-layer LSTM (128→32 units)
- **Attention**: Bahdanau additive attention mechanism
- **Decoder**: Single-layer LSTM (32 units) with attention context
- **Bottleneck**: Mean-pooled encoder states
- **Loss**: MSE reconstruction error

### 3. Anomaly Transformer
- **Architecture**: 8-layer transformer encoder (d_model=512, 8 heads)
- **Innovation**: Learnable Gaussian-prior attention per head
- **Scoring**: Symmetrized KL divergence between series and prior attention
- **Loss**: MSE reconstruction + λ×discrepancy loss (λ=0.1)

## 🛠️ Installation & Setup

### Requirements
```bash
# Core dependencies
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
pathlib
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd TimeSeries-Anomaly-Detection

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm

# Run the analysis
cd notebook/
jupyter notebook EDA.ipynb                    # Data exploration
jupyter notebook LSTM_Attention.ipynb         # LSTM with attention
jupyter notebook LSTM_VAE.ipynb              # Variational autoencoder  
jupyter notebook Transformer.ipynb           # Anomaly transformer
```

## 📁 Project Structure

```
TimeSeries-Anomaly-Detection/
├── README.md                          # This file
├── Assignment_2_requirements.pdf      # Original assignment description
├── Assignment_2_Report.pdf           # Detailed academic report
├── data/                             # Dataset directory
│   ├── train/                        # Training data (.npy files)
│   ├── test/                         # Test data (.npy files) 
│   ├── 2018-05-19_15.00.10/         # Model outputs
│   └── labeled_anomalies.csv         # Ground truth annotations
└── notebook/                         # Implementation notebooks
    ├── EDA.ipynb                     # Exploratory data analysis
    ├── LSTM_Attention.ipynb          # LSTM with Bahdanau attention
    ├── LSTM_VAE.ipynb               # LSTM Variational Autoencoder
    └── Transformer.ipynb            # Anomaly Transformer implementation
```

## 🔬 Methodology

### Preprocessing Pipeline
1. **Quality Screening**: Remove NaN values (<0.01%), clip extreme outliers (±5σ)
2. **Normalization**: Z-score standardization (zero mean, unit variance)
3. **Segmentation**: Sliding window approach
   - LSTM models: Window size=10, stride=5 (50% overlap)
   - Transformer: Window size=200, stride=50 
4. **Label Alignment**: Window-level binary labels based on overlap with anomaly spans
5. **Quality Control**: Filter channels with <100 anomalous windows
6. **Split**: 80% train, 20% validation (normal only); all anomalies in test set

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Batch Size**: 32
- **Scheduler**: ReduceLROnPlateau (factor=0.75, patience=5)
- **Early Stopping**: 10 epochs patience
- **Threshold**: 70th percentile of training scores

### Evaluation Metrics
- **Classification**: Precision, Recall, F1-score, Accuracy
- **Probabilistic**: AUROC, AUPRC (handles class imbalance)
- **Visualization**: ROC curves, Precision-Recall curves, confusion matrices
- **Reporting**: Macro-averaged across 28 channels

## 📈 Performance Analysis

### Channel-Specific Performance
**High-Performing Channels (F1 > 0.80):**
- D-1, D-3, D-4, D-9, D-12: Clear anomaly signatures, abrupt changes
- Consistent across all architectures

**Challenging Channels (F1 < 0.30):**
- A-8, E-series channels: Quasi-periodic patterns, high noise
- Require adaptive thresholding or specialized preprocessing

### Computational Efficiency
| Model | Parameters | Training Time | GPU Memory | Inference Speed |
|-------|------------|---------------|------------|-----------------|
| LSTM-VAE | ~500K | 2 hours | 3.5 GB | Fast |
| LSTM-Attention | ~350K | 2 hours | 3.5 GB | **Fastest** |
| Anomaly Transformer | ~2M | 6 hours | 6.8 GB | Moderate |

### Trade-off Analysis
- **Precision-Critical**: Choose LSTM-VAE (best F1 and AUROC)
- **Resource-Constrained**: Choose LSTM-Attention (smallest, fastest)
- **Recall-Critical**: Choose Anomaly Transformer (highest sensitivity)

## 🎯 Use Cases & Applications

### Industrial Applications
- **Spacecraft Telemetry**: Real-time fault detection in mission-critical systems
- **Manufacturing**: Equipment health monitoring and predictive maintenance  
- **IoT Networks**: Sensor anomaly detection in smart infrastructure
- **Financial Systems**: Fraud detection in transaction streams

### Deployment Recommendations
- **Edge Computing**: LSTM-Attention (low resource footprint)
- **Cloud Processing**: LSTM-VAE (best accuracy-efficiency balance)
- **Safety-Critical**: Anomaly Transformer (maximizes anomaly capture)

## 🚀 Future Improvements

### Model Enhancements
- **Adaptive Thresholding**: Dynamic, per-channel threshold adjustment
- **Cross-Channel Correlation**: Model inter-sensor dependencies
- **Continual Learning**: Adapt to concept drift over time

### Technical Extensions  
- **Real-Time Processing**: Streaming anomaly detection pipeline
- **Ensemble Methods**: Combine multiple model predictions
- **Explainable AI**: Attention visualization and feature attribution
- **Federated Learning**: Distributed training across multiple spacecraft

## 📚 References

1. Hundman, K. et al. "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding." KDD 2018.
2. Bahdanau, D. et al. "Neural machine translation by jointly learning to align and translate." ICLR 2015.
3. Kingma, D.P. & Welling, M. "Auto-encoding variational bayes." ICLR 2014.
4. Park, W.H. et al. "Anomaly Transformer: Time-series anomaly detection with association discrepancy." NeurIPS 2022.

## 👤 Author

**Sohan Arun**  
Department of Computer Science  
Blekinge Institute of Technology  
📧 soar24@student.bth.se


---

*For detailed technical analysis, experimental setup, and theoretical background, please refer to the complete research report: `Assignment_2_Report.pdf`*