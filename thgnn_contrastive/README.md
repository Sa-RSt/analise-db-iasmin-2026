# THGNN Replication (C-MAPSS)

This directory contains a replication of the paper:
**"Temporal and Heterogeneous Graph Neural Network for Remaining Useful Life Prediction"** (Wen et al., 2025).

## Architecture
- **Shared GRU Encoder**: Captures temporal dynamics for each sensor.
- **FiLM-based Heterogeneity Adaptation**: A hypernetwork ($\tau$) modulates a global mapping function ($f$) based on sensor signals and type embeddings, accounting for sensor heterogeneity.
- **Heterogeneous Sensor Graph**: Dynamic graph construction using cosine similarity of modulated representations at each time step.
- **Temporal Message Passing**: Spatial correlation capture across sensors.
- **Hawkes-style Aggregation**: Time-decaying aggregation of historical neighbor information.

## Project Structure
- `preprocess.py`: CMAPSS data loading, normalization, and window generation.
- `model.py`: THGNN model implementation in PyTorch.
- `train.py`: Training script for FD001-FD004.
- `test.py`: Evaluation script for the trained models.
- `scoring.py`: RUL evaluation metrics (RMSE and CMAPSS Score).

## Usage
1. Ensure `CMAPSSData/` is available in the project root.
2. Run training: `python thgnn_replication/train.py`
3. Run testing: `python thgnn_replication/test.py`

## Notes
- The replication uses $L=50$ and $d=16$ as suggested by the paper's sensitivity analysis for C-MAPSS.
- Piecewise RUL is capped at 125.
- The 14 active sensors (excluding constant ones) are mapped to 5 functional types (Temperature, Pressure, Speed, etc.).
