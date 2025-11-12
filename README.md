# IMDB Sentiment Classification with RNN, LSTM, and BiLSTM

A comprehensive study comparing recurrent neural network architectures for binary sentiment classification on the IMDB movie review dataset. This project systematically evaluates 162 different configurations across three model architectures, three activation functions, three optimizers, three sequence lengths, and two gradient clipping settings.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Experimental Setup](#experimental-setup)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Reproducibility](#reproducibility)

## 🎯 Project Overview

This project implements and compares three recurrent neural network architectures for sentiment classification:

- **RNN**: Vanilla recurrent neural network
- **LSTM**: Long Short-Term Memory network
- **BiLSTM**: Bidirectional LSTM

The study explores how different hyperparameters affect model performance:
- **Activation functions**: tanh, ReLU, sigmoid
- **Optimizers**: Adam, SGD, RMSprop
- **Sequence lengths**: 25, 50, 100 tokens
- **Gradient clipping**: Enabled (1.0) or disabled

### Total Experiments: 162
3 models × 3 activations × 3 optimizers × 3 sequence lengths × 2 gradient clipping options = 162 configurations

## 🏆 Key Findings

### Best Configuration
- **Model**: BiLSTM
- **Activation**: tanh
- **Optimizer**: Adam
- **Sequence Length**: 100
- **Gradient Clipping**: No
- **Performance**: 82.35% accuracy, 0.8235 F1-score

### Model Performance Summary
| Model   | Average Accuracy | Average F1-Score |
|:--------|-----------------:|-----------------:|
| BiLSTM  | 67.02%           | 0.6535           |
| LSTM    | 65.61%           | 0.6340           |
| RNN     | 57.85%           | 0.5677           |

### Key Insights
1. **BiLSTM outperforms** both LSTM and RNN, capturing bidirectional context for better sentiment understanding
2. **tanh activation** provides smoother training and better final performance compared to ReLU and sigmoid
3. **Adam optimizer** consistently provides faster and more stable convergence
4. **Longer sequences (100)** perform best, suggesting longer context improves sentiment classification
5. **Gradient clipping** helps with RNN stability but doesn't improve BiLSTM performance

## 📊 Dataset

- **Source**: IMDB Dataset (50,000 movie reviews)
- **Classes**: Binary (positive/negative) - balanced 25,000 each
- **Preprocessing**:
  - Text cleaning (lowercase, HTML removal, punctuation handling)
  - Tokenization using Keras Tokenizer
  - Vocabulary size: 10,000 most frequent words
  - Sequence padding/truncation to fixed lengths
- **Split**: 50/50 train-test split (stratified, random_state=42)
- **Average review length**: ~279 words (median: 209 words)

## 🏗️ Architecture

All models share the following base architecture:
- **Embedding layer**: 100-dimensional embeddings
- **RNN/LSTM layers**: 2 layers, 64 hidden units
- **Dropout**: 0.5
- **Output**: Single sigmoid output for binary classification
- **Loss function**: BCEWithLogitsLoss

### Model Variants
- **RNNClassifier**: Standard RNN with tanh/ReLU/sigmoid activation
- **LSTMClassifier**: LSTM with gating mechanisms
- **BiLSTMClassifier**: Bidirectional LSTM (captures forward and backward context)

## 🔬 Experimental Setup

### Hyperparameters
- **Epochs**: 5 per configuration
- **Batch size**: 128
- **Learning rate**: 1e-3 (for all optimizers)
- **SGD momentum**: 0.9
- **Gradient clipping**: 1.0 (when enabled)

### Training Details
- All models trained with fixed random seed (42) for reproducibility
- GPU acceleration recommended (CUDA)
- Training progress logged per epoch (loss, accuracy, F1-score)
- Model checkpoints saved after each experiment

## 📁 Directory Structure

```
project-root/
├── src/                      # Source code modules
│   ├── preprocess.py         # Data preprocessing and tokenization
│   ├── models.py             # RNN, LSTM, BiLSTM model definitions
│   ├── train.py              # Training and evaluation functions
│   ├── evaluate.py           # Metrics calculation and plotting
│   └── utils.py              # Dataset and DataLoader utilities
├── results/                   # All output artifacts
│   ├── model_*.pt            # Saved PyTorch model weights (162 models)
│   ├── metrics.csv            # Comprehensive metrics for all runs
│   ├── summary_by_model.csv  # Aggregated metrics by model type
│   ├── best_config.csv       # Best performing configuration
│   ├── tokenizer.pkl         # Saved tokenizer for inference
│   ├── X_train_seq*.npy      # Preprocessed training sequences
│   ├── X_test_seq*.npy       # Preprocessed test sequences
│   ├── y_train_seq*.npy      # Training labels
│   ├── y_test_seq*.npy       # Test labels
│   ├── plots/                # Visualization figures
│   │   ├── f1_by_model.png
│   │   ├── accuracy_by_model.png
│   │   ├── f1_by_seqlen_model.png
│   │   └── gradclip_effect.png
│   └── Other Results/        # Additional outputs and sample predictions
├── Homework_3.ipynb          # Main Jupyter notebook (complete workflow)
├── main.ipynb                # Alternative notebook
├── IMDB Dataset.csv          # Raw dataset (50,000 reviews)
├── report.pdf                # Detailed project report
└── README.md                 # This file
```

## 🚀 Setup Instructions

### Requirements
- **Python**: 3.8 or newer (tested on 3.10)
- **CUDA**: Optional but recommended for GPU acceleration

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Homework 3"
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tensorflow keras  # For preprocessing tokenizer
pip install nltk tqdm
```

Or create a `requirements.txt` with:
```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorflow>=2.13.0
nltk>=3.8.0
tqdm>=4.65.0
```

Then install:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
```

## 💻 Usage

### Option 1: Interactive Notebook (Recommended)

1. Open `Homework_3.ipynb` in Jupyter Notebook or Google Colab
2. Execute cells sequentially:
   - **Cell 1-2**: Load and explore dataset
   - **Cell 3-4**: Preprocess data and create tokenizer
   - **Cell 5-6**: Convert to sequences and save arrays
   - **Cell 7-9**: Define models and dataset utilities
   - **Cell 10-11**: Training functions
   - **Cell 12-15**: Run experiments (full grid or subset)
   - **Cell 16-17**: Generate plots and summaries

The notebook will automatically:
- Preprocess the IMDB dataset
- Train all 162 configurations
- Save model weights and metrics
- Generate visualization plots
- Create summary reports

### Option 2: Command-Line Scripts

#### Preprocess Data
```bash
python src/preprocess.py --input "IMDB Dataset.csv" --output_dir results/
```

#### Train a Single Configuration
```bash
python src/train.py \
    --model LSTM \
    --activation tanh \
    --optimizer adam \
    --seq_len 100 \
    --clip_grad no \
    --epochs 5 \
    --batch_size 128
```

#### Evaluate and Plot Results
```bash
python src/evaluate.py --metrics results/metrics.csv --output_dir results/plots/
```

### Option 3: Python API

```python
from src.models import LSTMClassifier
from src.train import train_and_evaluate
from src.utils import make_loaders

# Load data
train_loader, test_loader = make_loaders(seq_len=100, batch_size=128)

# Train model
history, acc, f1 = train_and_evaluate(
    model_class=LSTMClassifier,
    model_name="LSTM",
    seq_len=100,
    activation="tanh",
    optimizer_name="adam",
    grad_clip=None,
    epochs=5,
    batch_size=128
)

print(f"Final Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
```

## 📈 Results

### Output Files

**Model Weights**: `results/model_<Model>_act<activation>_opt<optimizer>_seq<length>.pt`
- One saved model per experiment (162 total)

**Metrics**: 
- `results/metrics.csv`: Complete results for all experiments
- `results/summary_by_model.csv`: Aggregated statistics by model type
- `results/best_config.csv`: Best performing configuration details

**Visualizations** (in `results/plots/`):
- `f1_by_model.png`: F1-score comparison across architectures
- `accuracy_by_model.png`: Accuracy comparison across architectures
- `f1_by_seqlen_model.png`: F1-score vs. sequence length by model
- `gradclip_effect.png`: Impact of gradient clipping on performance

### Expected Runtime

- **Single configuration** (5 epochs, seq_len=100, GPU): ~5 minutes
- **Full grid** (162 configs, 5 epochs each): ~3-4 hours on Google Colab T4 GPU
- **Evaluation/plotting**: <2 minutes

## 🔄 Reproducibility

All experiments use fixed random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python random: `random.seed(42)`
- Train-test split: `random_state=42`, stratified

To exactly reproduce results:
1. Use Python 3.8+ (tested on 3.10)
2. Install exact package versions from `requirements.txt`
3. Use the same random seed settings
4. Run on the same hardware (GPU recommended)

## 📚 Additional Resources

- **Detailed Report**: See `report.pdf` for comprehensive analysis
- **Assignment PDF**: See `Homework 3.pdf` for original requirements
- **Notebook Comments**: Detailed explanations in `Homework_3.ipynb` cells

## 🤝 Contributing

This is a course project. For questions or issues:
- Check notebook comments and script docstrings
- Review the detailed report in `report.pdf`
- See code documentation in `src/` modules

## 📝 License

This project is for educational purposes as part of a Data641 NLP course assignment.

---

**Note**: The full experiment grid (162 configurations) requires significant computational resources. For quick testing, modify the experiment grid in the notebook to run a subset of configurations.
