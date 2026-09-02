# Implementation Plan: Physics-Informed, Rotation-Invariant AI Velocity Estimator

Integrate all 5 deep learning strategies into the core codebase to maximize generalization across unseen trips and mount orientations without overfitting or cheating.

## User Review Required

> [!IMPORTANT]
> - **Zero Data Leakage**: The training process will strictly use `S-S1` and `S-M` for training, and evaluate validation on the completely unseen 2.5-hour trip `S-S2`.
> - **Self-Contained Execution**: All code changes will be saved to the repository files, and we will provide the exact commands for you to run training and benchmarking directly in your terminal.

---

## Proposed Changes

### 1. Dataset Layer: Rotation Augmentation & Vectorized Striding
#### [MODIFY] [sih/models/dataset.py](file:///c:/Users/carpe/SIH/sih/models/dataset.py)
- Vectorize sliding window creation using NumPy memory strides for fast loading (<1s).
- Add online 3D $\mathrm{SO}(3)$ rotation augmentation ($\mathbf{a}' = \mathbf{R}\mathbf{a}, \boldsymbol{\omega}' = \mathbf{R}\boldsymbol{\omega}$) during training to make the network mount-invariant.
- Add sensor noise jitter ($\sigma_a = 0.02\text{ m/s}^2, \sigma_\omega = 0.005\text{ rad/s}$).
- Support fine-grained striding (`step_size=2`, 0.2s) expanding training data to 78,762 unique windows.

---

### 2. Model Layer: Multi-Scale Dilated TCN ($d \in \{1, 2, 4, 8, 16\}$)
#### [MODIFY] [sih/models/tcn_attention.py](file:///c:/Users/carpe/SIH/sih/models/tcn_attention.py)
- Expand the temporal receptive field with 4 dilated stages ($d=1, 2, 4, 8, 16$).
- Implement Huber-regularized Gaussian NLL loss with kinematic consistency penalty.

---

### 3. Training Script: Unified Multi-Strategy Pipeline
#### [MODIFY] [train_velocity_model.py](file:///c:/Users/carpe/SIH/train_velocity_model.py)
- Integrate Cosine Annealing with Warm Restarts.
- Log training NLL, validation RMSE (m/s and km/h), and MAE live with progress bars.
- Automatically save best checkpoint to `models/checkpoints/best_velocity_model.pt` and plot curves to `artifacts/training_curves.png`.

---

## Verification Plan

### Automated Execution by User
1. **Train the AI Model**:
   ```powershell
   python train_velocity_model.py --epochs 35 --batch_size 128 --step_size 2
   ```
2. **Run the Multi-Trip Benchmark**:
   ```powershell
   python benchmarks/run_phase3_ai_fusion.py
   ```
3. **Inspect Output Plots**:
   ```powershell
   start artifacts/S-S1_60s_blackout_at_300s_phase_3_es-ekf_plus_ai_velocity_tcn-attention.png
   start artifacts/S-S2_S-S2_30s_blackout_at_120s_phase_3_es-ekf_plus_ai_velocity_tcn-attention.png
   ```
