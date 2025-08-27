# TATAttention

Deep learning (CNN + Multi-Head Self-Attention) for **counting TATA-box motifs** in eukaryotic promoter sequences.

## What counts as a “motif”
We count **every occurrence** of:
1. **Exact** `TATAAA` on the plus strand.
2. **Single-mismatch** variants of `TATAAA` on the plus strand (e.g., `TAGAAA`, `TATCAA`, etc.; Hamming distance = 1).
3. All of the above also on the **reverse-complementary** strand.

The label for each sequence is the total number of such occurrences.

## Model architecture
The neural network architecture is:

- **Stem block**: 1D convolution (kernel=7) + batch norm + ReLU + dropout.  
- **Residual block**: two 1D convolutions (kernel=5) with skip connection, batch norm, ReLU, and dropout.  
- **Multi-Head Self-Attention**: attends over the sequence representation (no masks, no positional encoding).  
- **Mean Pooling**: averages representations across sequence positions.  
- **Linear head**: single regression output predicting the total motif count.

## Project structure
- `src/` — code (data collection, one-hot, loaders, model, train/eval)
- `data/` — input data (not uploaded to repo; keep only small samples or `README.md`)
- `checkpoints/` — trained weights (ignored by git)
- `README.md`, `.gitignore`

## Data & preprocessing
- Inputs are fixed-length sequences (e.g., L=600) encoded as **5 channels** (`A,C,G,T,N/mask`).
- Labels are total motif counts as defined above (plus strand + reverse complement; exact + 1-mismatch).
- See `src/data/GenomicDataCollecting.py`, `src/data/one_hot.py`, `src/data/makeLoaders.py`.

## How to run
python -m src.main
# or
python -m src.train_eval

## Requirements
torch
numpy
pandas
scikit-learn
pyyaml
