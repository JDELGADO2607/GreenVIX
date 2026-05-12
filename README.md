# GJR-GARCH + GRU Synthetic VIX Pipeline
## Technical Documentation — v2 (Walk-Forward)

> **File:** `gjr_garch_gru_synthetic_vix.R`  
> **Language:** R (≥ 4.2)  
> **Last updated:** 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Academic Basis](#2-academic-basis)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Dependencies](#4-dependencies)
5. [Configuration Reference](#5-configuration-reference)
6. [Data Requirements](#6-data-requirements)
7. [Function Reference](#7-function-reference)
8. [Pipeline Steps](#8-pipeline-steps)
9. [Walk-Forward Validation Protocol](#9-walk-forward-validation-protocol)
10. [Transfer Learning Mechanism](#10-transfer-learning-mechanism)
11. [Output Files](#11-output-files)
12. [Performance Metrics](#12-performance-metrics)
13. [Methodological Optimizations (v1 → v2)](#13-methodological-optimizations-v1--v2)
14. [Error Handling & Fallback Chains](#14-error-handling--fallback-chains)
15. [Extending the Pipeline](#15-extending-the-pipeline)
16. [Known Limitations](#16-known-limitations)
17. [References](#17-references)

---

## 1. Overview

This pipeline constructs a **synthetic VIX** (implied volatility index) for an illiquid ESG index by exploiting the *universal volatility formation mechanism* — the empirical finding that the non-linear mapping from physical conditional variance to risk-neutral implied volatility is asset-invariant and therefore transferable.

The method proceeds in two stages:

**Stage 1 — Source domain (S&P 500):** A GJR-GARCH(1,1) model extracts daily physical conditional variance from SPX log-returns. A stacked GRU neural network then learns the non-linear mapping from `[r_spx_t, f(h_spx_t)]` to `VIX_t` using walk-forward cross-validation to ensure regime robustness.

**Stage 2 — Target domain (ESG index):** An identical GJR-GARCH specification is fitted to the ESG log-returns. The ESG features are normalised using the SPX scalers (cross-domain normalisation), then fed into the pre-trained GRU to produce the synthetic VIX time series.

### What this pipeline is NOT

- It does not price individual options or compute a term structure.
- It does not assume the ESG index has liquid traded options.
- It does not guarantee arbitrage-free implied volatility surfaces.
- It is not a Shiny app or interactive dashboard.

---

## 2. Academic Basis

| Concept | Reference |
|---|---|
| Asymmetric GARCH with leverage effect | Glosten, Jagannathan & Runkle (1993) |
| Closed-form GARCH option pricing (H-N model) | Heston & Nandi (2000) |
| Universal volatility formation mechanism | Ruan, Zhang & Luo (2022) |
| Gated Recurrent Units (GRU) | Cho, van Merrienboer et al. (2014) |
| GRU vs LSTM for financial time series | Bgates & Nouri (2021) |
| Walk-forward validation for time series ML | Cerqueira, Torgo & Mozetič (2020) |

### The Universal Volatility Mechanism Hypothesis

The hypothesis states that the conditional transformation:

```
Φ : (r_t, h_t^physical) → σ_t^implied
```

is governed by a universal non-linear mapping that can be learned from any liquid asset with observable implied volatility (here: S&P 500 / CBOE VIX) and transferred to any other asset whose physical variance dynamics are structurally similar.

The key assumption is that the **functional form** of Φ is invariant across assets, even if the *scale* and *persistence* of the variance processes differ. This is operationalised by the cross-domain normalisation step described in [Section 10](#10-transfer-learning-mechanism).

### GJR-GARCH as Heston-Nandi Substitute

The original Heston-Nandi (2000) model is:

```
h_t = ω + α(ε_{t-1} − γ√h_{t-1})² + β·h_{t-1}
```

This is structurally equivalent to the GJR-GARCH(1,1):

```
h_t = ω + (α + γ·I_{t-1})·ε_{t-1}² + β·h_{t-1}
```

where `I_{t-1} = 1` if `ε_{t-1} < 0` (bad news). Both models capture the **leverage effect** — the empirical asymmetry whereby negative return shocks increase future variance by more than positive shocks of equal magnitude. The `fOptions` R package implementing the H-N closed form is deprecated; `rugarch::ugarchfit()` with `model = "gjrGARCH"` is the academically accepted substitute.

The stationarity condition for GJR-GARCH is:

```
α + γ/2 + β < 1
```

which the script checks and reports as the *persistence* diagnostic.

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SOURCE DOMAIN (SPX)                         │
│                                                                     │
│  SPX prices ──► diff(log(.)) ──► r_spx                             │
│                                      │                              │
│                                      ▼                              │
│                           GJR-GARCH(1,1) fit                        │
│                           [tol=1e-12, delta=1e-11]                  │
│                                      │                              │
│                                      ▼                              │
│                             h_spx_raw (variance)                    │
│                                      │                              │
│                          APPLY_VAR_TRANSFORM (sqrt)                 │
│                                      │                              │
│                                      ▼                              │
│                   fh_spx = sqrt(h_spx)  [cond. SD]                  │
│                                      │                              │
│   r_spx ──┐                          │                              │
│           ├──► SCALE_FIT ──► X_spx_scaled (N × 2)                  │
│  fh_spx ──┘                          │                              │
│                                      ▼                              │
│                          BUILD_SEQUENCES (T=20)                     │
│                         3-D tensor (N-T, 20, 2)                     │
│                                      │                              │
│              ┌───────────────────────┘                              │
│              │                                                      │
│              ▼           WALK-FORWARD LOOP (k folds)                │
│     ┌────────────────┐   ┌──────────────────────────┐              │
│     │  Fold Window k │──►│  BUILD_GRU_MODEL()        │              │
│     │  Train: 252d   │   │  GRU(64)→Drop→GRU(32)    │              │
│     │  Pred:   21d   │   │  →Dense(16)→Dense(1)      │              │
│     └────────────────┘   └──────────┬───────────────┘              │
│                                     │ OOS predictions               │
│                                     ▼                               │
│                          wf_results_df (all folds)                  │
│                                                                     │
│              ▼  FINAL GRU retrained on full SPX history             │
│     ┌────────────────────────────────────────────────────┐         │
│     │  final_gru  (frozen weights after training)        │         │
│     └────────────────────────┬───────────────────────────┘         │
└────────────────────────────────────────────────────────────────────-┘
                                │  Transfer
                                │  (weights frozen,
                                │   SPX scalers applied to ESG)
┌───────────────────────────────▼─────────────────────────────────────┐
│                        TARGET DOMAIN (ESG)                          │
│                                                                     │
│  ESG prices ──► diff(log(.)) ──► log_returns                       │
│                                        │                            │
│                                        ▼                            │
│                             GJR-GARCH(1,1) fit                      │
│                             (same spec as SPX)                      │
│                                        │                            │
│                                        ▼                            │
│                               h_esg_raw                             │
│                                        │                            │
│                            APPLY_VAR_TRANSFORM (sqrt)               │
│                                        │                            │
│                                        ▼                            │
│                  fh_esg = sqrt(h_esg)                               │
│                                        │                            │
│  log_returns ──┐                       │                            │
│                ├──► SPX scalers ──► X_esg_scaled (M × 2)           │
│      fh_esg ───┘   (cross-domain normalisation)                     │
│                                        │                            │
│                            BUILD_SEQUENCES (T=20)                   │
│                           3-D tensor (M-T, 20, 2)                   │
│                                        │                            │
│                                        ▼                            │
│                         GRU_INFER(final_gru, X_esg)                 │
│                                        │                            │
│                                        ▼                            │
│                    synth_vix = scaler_vix$inverse(...)              │
│                                        │                            │
│                                        ▼                            │
│                          synthetic_vix_df  ──► CSV                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dependencies

All packages are auto-installed if missing via the bootstrap loop at the top of the script.

| Package | Version | Role |
|---|---|---|
| `readxl` | any | Excel ingestion of the `.xlsx` data file |
| `rugarch` | ≥ 1.4 | GJR-GARCH and eGARCH specification and MLE fitting |
| `keras` | ≥ 3.0 (keras3) | High-level GRU/Dense layer API and training loop |
| `tensorflow` | ≥ 2.12 | TensorFlow backend; provides `tf$constant()` for inference |
| `ggplot2` | ≥ 3.4 | All diagnostic and output plots |
| `dplyr` | ≥ 1.1 | `mutate()` used in fold-shading plot data prep |
| `tidyr` | ≥ 1.3 | `pivot_longer()` for plot reshaping (optional use) |
| `zoo` | any | Rolling utilities and NA handling helpers |
| `scales` | any | Axis formatting in ggplot2 |
| `Metrics` | any | RMSE / MAE helper functions |
| `gridExtra` | optional | Two-panel GARCH volatility plot (graceful fallback if absent) |

### Python / TensorFlow backend setup

Run once before first execution (in an R interactive session):

```r
keras::install_keras(method = "conda", envname = "r-keras")
tensorflow::install_tensorflow(envname = "r-keras")
```

On Kaggle, the TensorFlow environment is pre-installed; no manual setup is required.

### keras3 compatibility notes

This script is written for **keras3** (the `keras` R package ≥ 3.0), which introduced breaking API changes from keras2:

| Deprecated (keras2) | Replacement (keras3) |
|---|---|
| `keras::predict(model, X)` | `model(tf$constant(X), training=FALSE)` |
| `model %>% compile(...)` | `keras::compile(model, ...)` |
| `model %>% fit(...)` | `keras::fit(model, ...)` |
| `save_model_hdf5(model, path)` | `save_model(model, path)` or Python passthrough |

---

## 5. Configuration Reference

All hyper-parameters are declared at the top of the script in a single consolidated block for easy tuning.

### GARCH parameters

| Variable | Default | Description |
|---|---|---|
| `GARCH_VARIANT` | `"gjrGARCH"` | GARCH model family. Alternatives: `"eGARCH"` |
| `GARCH_DIST` | `"sstd"` | Error distribution. `"sstd"` = skewed Student-t, handles fat tails and skewness simultaneously. Alternatives: `"std"`, `"norm"`, `"ged"` |
| `GARCH_TOL` | `1e-12` | **[OPT-3]** Convergence tolerance for the log-likelihood gradient norm. Tighter than rugarch default (`1e-8`) to prevent premature convergence on flat likelihood surfaces near high-persistence regimes. |
| `GARCH_DELTA` | `1e-11` | **[OPT-3]** Finite-difference step for numerical Hessian approximation used in standard error computation. |

### Variance feature transform

| Variable | Default | Options | Description |
|---|---|---|---|
| `VAR_TRANSFORM` | `"sqrt"` | `"sqrt"`, `"log"`, `"none"` | **[OPT-2]** Transformation applied to raw conditional variance `h_t` before Min-Max scaling. `"sqrt"` = conditional SD. `"log"` = log-variance. `"none"` = raw (not recommended for neural networks). |

### Sequence parameters

| Variable | Default | Description |
|---|---|---|
| `TIMESTEPS` | `20L` | Lookback window length in trading days (~1 calendar month). Input tensor shape becomes `(samples, 20, 2)`. |
| `N_FEATURES` | `2L` | Number of input features per timestep: `[r_t, f(h_t)]`. |

### Walk-forward parameters

| Variable | Default | Description |
|---|---|---|
| `TRAIN_DAYS` | `252L` | **[OPT-4]** Training window size per fold in trading days (~1 year). |
| `PRED_DAYS` | `21L` | **[OPT-4]** Prediction horizon per fold in trading days (~1 month). |
| `MIN_FOLDS` | `3L` | Minimum number of complete folds required. The script aborts with an informative error if fewer folds are available. |

### GRU / training parameters

| Variable | Default | Description |
|---|---|---|
| `GRU_UNITS` | `64L` | Units in the first GRU layer. Second layer uses `GRU_UNITS / 2 = 32`. |
| `DROPOUT` | `0.20` | Dropout rate applied after each GRU layer. Disabled at inference (`training=FALSE`). |
| `LR` | `1e-3` | Initial Adam learning rate. Decayed by `PATIENCE_LR` callback. |
| `BATCH_SIZE` | `32L` | Mini-batch size. Must be integer. |
| `EPOCHS` | `100L` | Maximum training epochs; early stopping typically terminates well before this. |
| `PATIENCE_STOP` | `15L` | Early stopping patience (epochs without val_loss improvement). |
| `PATIENCE_LR` | `7L` | LR reduction patience. When triggered, LR is halved (factor = 0.5). |
| `L2_REG` | `1e-4` | L2 weight regularisation coefficient applied to both GRU kernel matrices. |

---

## 6. Data Requirements

### Input file

```
DATA_PATH <- "/kaggle/input/datasets/robertosnotetaker/base-vix-verde/Base Definitiva VIX Verde.xlsx"
```

Adjust `DATA_PATH` to point to your local or cloud copy of the Excel file.

### Required columns

| Column | Type | Description |
|---|---|---|
| `Date` | Date/POSIXct | Trading calendar dates. Used for alignment and plot axes. |
| `SPX` | Numeric | S&P 500 closing price levels. Log-returns are computed internally. |
| `VIX` | Numeric | CBOE VIX index (implied volatility, in index points). This is the training target. |
| `Price (Adjusted BESG)` | Numeric | Adjusted ESG index price levels. Log-returns computed internally. |

### Minimum data requirements

| Series | Minimum length | Reason |
|---|---|---|
| SPX / VIX | `TRAIN_DAYS + PRED_DAYS × MIN_FOLDS + TIMESTEPS` ≈ 357 obs | To support at least `MIN_FOLDS = 3` walk-forward folds |
| ESG | `TIMESTEPS + 1` = 21 obs | Minimum for one complete sequence (inference only) |

The script enforces these with `stopifnot()` assertions that produce informative errors on failure.

### Pre-processing performed by the script

The following transformations are applied internally — you do **not** need to pre-process the raw price data:

```r
r_spx       <- diff(log(spx))          # SPX log-returns
log_returns <- diff(log(esg_raw))      # ESG log-returns
vix_aligned <- vix[2:length(vix)]      # VIX aligned to returns (drop first obs)
```

---

## 7. Function Reference

### `FIT_GARCH(returns, label, garch_variant, dist, tol, delta)`

Fits a GJR-GARCH(1,1) model via maximum likelihood estimation using `rugarch::ugarchfit()`.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `returns` | — | Numeric vector of log-returns |
| `label` | `"Series"` | Label string for console messages |
| `garch_variant` | `GARCH_VARIANT` | `"gjrGARCH"` or `"eGARCH"` |
| `dist` | `GARCH_DIST` | Error distribution |
| `tol` | `GARCH_TOL` | Likelihood convergence tolerance |
| `delta` | `GARCH_DELTA` | Hessian finite-difference step |

**Returns:** Named list with:
- `fit` — `uGARCHfit` S4 object from rugarch
- `h_t` — Numeric vector of daily conditional **variances** (`sigma(fit)^2`)
- `spec` — `uGARCHspec` specification object (reusable for forecasting)
- `persistence` — Scalar: `α + γ/2 + β`

**Solver fallback chain:**

```
hybrid [tol=1e-12] ──► nlminb [tol=1e-12] ──► solnp [loose tol]
```

Each fallback fires only on `tryCatch` error from the preceding solver.

---

### `APPLY_VAR_TRANSFORM(h, method)`

Applies the configured variance feature transformation.

| `method` | Formula | Use case |
|---|---|---|
| `"sqrt"` | `f(h) = √h` | Moderate spikes; preserves volatility units |
| `"log"` | `f(h) = log(h + ε)` | Strong linearisation; removes absolute scale |
| `"none"` | `f(h) = h` | Passthrough (not recommended) |

**Guard:** `log` mode adds `1e-12` before taking logs to prevent `log(0) = -Inf`.

---

### `INVERT_VAR_TRANSFORM(fh, method)`

Exact inverse of `APPLY_VAR_TRANSFORM`. Used for diagnostic back-transforms.

| `method` | Inverse formula |
|---|---|
| `"sqrt"` | `h = fh²` |
| `"log"` | `h = exp(fh) − ε` |
| `"none"` | `h = fh` |

---

### `SCALE_FIT(x)`

Fits a Min-Max scaler on vector `x` and returns a closure list.

**Returns:** Named list with:
- `min`, `max` — Observed range
- `transform(v)` — Maps `v` to `[0, 1]`: `(v − min) / (max − min + ε)`
- `inverse(v)` — Exact inverse: `v × (max − min + ε) + min`

The `ε = 1e-10` term prevents division by zero for constant series.

---

### `BUILD_SEQUENCES(X, y, timesteps)`

Constructs the 3-D sliding-window tensor required by `layer_gru()`.

**Algorithm:**

```
For i in 1 .. (nrow(X) - timesteps):
    X_seq[i, , ] ← X[i : i + timesteps - 1, ]   # (T × F) input window
    y_seq[i]     ← y[i + timesteps]               # next-step scalar target
```

**Returns:**
- `X` — 3-D array `(n_samples, timesteps, n_features)`, dtype `double`
- `y` — Numeric vector `(n_samples,)`
- `n` — Integer count of valid samples

**Edge case:** Returns `list(X=NULL, y=NULL, n=0L)` if `nrow(X) <= timesteps`.

---

### `BUILD_GRU_MODEL(timesteps, n_features, units, dropout, lr, l2)`

Constructs and compiles a stacked GRU model using the keras3 functional API.

**Architecture:**

```
Input (TIMESTEPS, N_FEATURES)
  └─► GRU(64, return_sequences=TRUE, L2=1e-4)    [GRU layer 1]
        └─► Dropout(0.20)
              └─► GRU(32, return_sequences=FALSE, L2=1e-4)  [GRU layer 2]
                    └─► Dropout(0.20)
                          └─► Dense(16, activation="relu")
                                └─► Dense(1, activation="linear")   [VIX output]
```

**GRU gate equations (per timestep):**

```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)          # reset gate
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)          # update gate
ñ_t = tanh(W_n · [r_t ⊙ h_{t-1}, x_t] + b_n) # candidate hidden state
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t       # new hidden state
```

**Compiled with:**
- Optimizer: Adam (`lr = 1e-3`)
- Loss: Mean Squared Error
- Metrics: Mean Absolute Error

**Note:** `keras::compile()` mutates the model in-place and returns `NULL` in keras3. The function explicitly returns the model object after calling compile.

---

### `GRU_INFER(model, X)`

Runs inference on a trained GRU model. keras3-compatible replacement for the removed `keras::predict()` generic.

```r
GRU_INFER <- function(model, X) {
  tensor <- tensorflow::tf$constant(X, dtype = "float32")
  as.vector(as.array(model(tensor, training = FALSE)))
}
```

`training = FALSE` is critical: it disables Dropout during inference. Without it, predictions are stochastic across calls.

---

## 8. Pipeline Steps

The script executes nine numbered stages, logged to console as `[N/9]`.

### Step 1 — Data loading `[1/9]`

Reads the Excel file, extracts the four raw series, computes log-returns via `diff(log(.))`, and aligns VIX to the return series (shifting by 1 to drop the observation lost by `diff()`).

**Key assertion:**
```r
stopifnot(length(r_spx) == length(vix_aligned))
stopifnot(length(log_returns) >= TRAIN_DAYS + PRED_DAYS + TIMESTEPS)
```

### Step 2 — SPX GJR-GARCH fit `[2/9]`

Fits GJR-GARCH(1,1) on the full SPX log-return series `r_spx`. Extracts `h_spx_raw` (daily conditional variance) and reports the persistence diagnostic.

### Step 3 — Variance feature transform `[3/9]`

Applies `APPLY_VAR_TRANSFORM(h_spx_raw)` to produce `fh_spx`. Reports skewness before and after to confirm linearisation. Fits Min-Max scalers `scaler_r_spx`, `scaler_fh_spx`, `scaler_vix` on the full SPX series.

### Step 4 — Walk-forward validation `[4/9]`

Executes the rolling-window GRU training loop. See [Section 9](#9-walk-forward-validation-protocol) for full protocol details.

### Step 5 — Final GRU refit `[5/9]`

After walk-forward evaluation, trains a fresh GRU on the **entire** SPX series (90% train / 10% internal validation). This is the model that will be used for ESG transfer. Early stopping and LR decay are active.

### Step 6 — ESG GJR-GARCH fit `[6/9]`

Fits the same GJR-GARCH specification (including `GARCH_DIST`, `GARCH_TOL`, `GARCH_DELTA`) on `log_returns`. Extracts `h_esg_raw` and reports ESG persistence.

### Step 7 — Transfer inference `[7/9]`

Applies `APPLY_VAR_TRANSFORM` to `h_esg_raw`, normalises using the SPX scalers, reshapes into 3-D sequences, and calls `GRU_INFER(final_gru, X_esg_tensor)`. Inverse-transforms the output using `scaler_vix$inverse()`. See [Section 10](#10-transfer-learning-mechanism) for design rationale.

### Step 8 — Plots and outputs `[8/9]`

Resolves `OUTPUT_DIR`, writes all seven PNG plots and four CSV files, attempts model save with three-level fallback.

### Step 9 — Verification `[9/9]`

Checks existence and file size of every expected output, prints the full performance summary table, and raises a `warning()` if any file is missing.

---

## 9. Walk-Forward Validation Protocol

### Motivation

A static 80/20 chronological split (as used in v1) has a critical flaw for financial time series: the single test window may coincide with an unusually calm or unusually volatile period, making the reported metrics non-representative of general performance. Walk-forward validation provides a distribution of OOS errors across multiple market regimes.

### Rolling window design

```
|←──── TRAIN_DAYS ────→|←─ PRED_DAYS ─→|
 Fold 1:  [1 ............. 252 | 253 ... 273]
 Fold 2:  [22 ............ 273 | 274 ... 294]   (window rolls by PRED_DAYS)
 Fold 3:  [43 ............ 294 | 295 ... 315]
 ...
```

- Window type: **fixed-size rolling** (not expanding). The training window size stays constant at `TRAIN_DAYS`. This ensures each fold sees the same volume of recent history.
- Step size: `PRED_DAYS` (21 trading days). The window rolls by one prediction horizon per fold.
- Total folds: `floor((n_total_seq − TRAIN_DAYS) / PRED_DAYS)`

### Per-fold training

For each fold `k`:

1. Extract tensor slices for training (`TRAIN_DAYS` sequences) and prediction (`PRED_DAYS` sequences) windows.
2. Reserve the last 15% of the training window as an internal validation set for early stopping.
3. Call `BUILD_GRU_MODEL()` — weights initialised from scratch.
4. Train with callbacks: `EarlyStopping(patience=15)` + `ReduceLROnPlateau(factor=0.5, patience=7)` + `ModelCheckpoint`.
5. Run `GRU_INFER()` on the prediction window.
6. Collect OOS predictions and compute fold-level RMSE, MAE, R².
7. Destroy the fold model and call `keras::k_clear_session()` + `gc()` to release GPU/CPU memory.

### Fine-tuning alternative

The script includes a commented block explaining how to replace step 3 with **warm-start fine-tuning**: instead of building a fresh model, restore weights from the previous fold's checkpoint and reduce LR by 10×. This is recommended when:
- The total dataset is short (< 500 trading days).
- Regimes are slow-moving (e.g., emerging market indices).

---

## 10. Transfer Learning Mechanism

### The cross-domain normalisation assumption

The GRU learns the function:

```
f : [r̃_t, f̃(h_t)] → VIX̃_t
```

where tildes denote Min-Max-scaled values. The scaler parameters (min, max) are fitted on the full **SPX** series.

When ESG features are fed into the same model, they are scaled using the **same SPX scaler parameters**:

```r
esg_r_scaled  <- scaler_r_spx$transform(log_returns)
esg_fh_scaled <- scaler_fh_spx$transform(fh_esg)
```

This is the operationalisation of the universal mechanism hypothesis: we assert that, in the normalised space `[0, 1]²`, the mapping to implied volatility is the same for SPX and ESG. If ESG returns or conditional SD fall outside the SPX training range, they will clip to values outside `[0, 1]` — a natural indicator that the transfer assumption is being stressed.

### Why this works (theoretical justification)

The variance feature transform `f(h_t) = √h_t` maps both series to conditional SD, which is the natural unit of volatility. After this transform, the Min-Max scaler maps both to the same relative position within their empirical volatility regime. The GRU then maps relative volatility level and recent returns to a relative implied volatility level, which the VIX inverse-scaler maps back to index points.

The validity of this approach rests on two assumptions:
1. The **rank ordering** of volatility regimes (calm / normal / stressed) is preserved across the two domains.
2. The VIX scaler range covers the range of synthetic VIX values the ESG index would plausibly produce.

If (2) is violated — e.g., the ESG index is systematically more volatile than SPX — the synthetic VIX will be clipped. In that case, fit a separate VIX scaler calibrated to the ESG domain's expected range.

---

## 11. Output Files

All files are written to `OUTPUT_DIR`, which resolves as:

| Environment | Path |
|---|---|
| Kaggle Notebook | `/kaggle/working` |
| Custom override | `getOption("output_dir")` |
| Local / other | `getwd()` (current working directory) |

### CSV outputs

| File | Columns | Description |
|---|---|---|
| `synthetic_vix_esg.csv` | `Date`, `Synthetic_VIX` | **Primary deliverable.** Daily synthetic VIX for the ESG index. |
| `walkforward_oos_predictions.csv` | `Date`, `Fold`, `Actual_VIX`, `Pred_VIX` | OOS predictions from the walk-forward loop, labelled by fold. |
| `walkforward_fold_metrics.csv` | `fold`, `train_start`, `train_end`, `pred_start`, `pred_end`, `best_epoch`, `val_mse`, `oos_rmse`, `oos_mae`, `oos_r2` | Per-fold performance summary. |
| `spx_full_sample_fit.csv` | `Date`, `Actual`, `Predicted` | Final GRU full-sample fit on the SPX domain. |

### PNG plots

| File | Content |
|---|---|
| `01_walkforward_fold_metrics.png` | Bar chart of OOS RMSE per fold with aggregate RMSE reference line. |
| `02_walkforward_oos_fit.png` | Time series of actual vs. predicted CBOE VIX across all OOS fold windows, with alternating fold shading. |
| `03_final_gru_training_history.png` | Train and validation MSE curves for the final GRU (full SPX refit). |
| `04_spx_full_gru_fit.png` | Full-sample SPX actual vs. GRU-fitted VIX. |
| `05_synthetic_vix_esg.png` | Synthetic VIX time series for the ESG index with ±15% shaded band. |
| `06_spx_garch_vol_transform.png` | Two-panel: (top) annualised SPX conditional volatility; (bottom) same series after variance transform. Requires `gridExtra`; falls back to top panel only. |
| `07_esg_garch_vol.png` | Annualised ESG conditional volatility from GJR-GARCH. |

### Model file

| File | Format | Description |
|---|---|---|
| `gjr_garch_gru_model.keras` | Native keras3 format | Full model: architecture + weights + optimizer state. Load with `keras::load_model()`. |
| `gjr_garch_gru_weights.*` | TF SavedModel weights | Fallback if full-model save fails. Requires architecture to be rebuilt before loading. |

---

## 12. Performance Metrics

The following metrics are computed and reported at multiple levels.

### Walk-forward OOS metrics (primary evaluation)

Computed by concatenating all fold OOS predictions:

```
RMSE = √( mean( (VIX_pred - VIX_actual)² ) )
MAE  =   mean( |VIX_pred - VIX_actual| )
R²   = 1 - SS_res / SS_tot
```

All values are in **VIX index points** (after inverse-scaling from `[0, 1]`).

### Per-fold metrics

Each fold reports its own RMSE, MAE, and R², along with `best_epoch` (early stopping epoch) and `val_mse` (internal validation MSE). These are written to `walkforward_fold_metrics.csv` and plotted in `01_walkforward_fold_metrics.png`.

Persistent regression across folds (rising RMSE) indicates non-stationary regime evolution that the 252-day window may not be capturing — consider increasing `TRAIN_DAYS` or switching to fine-tuning mode.

### Final GRU full-sample metrics

Reported separately as in-sample metrics on the final model trained on the full SPX history. These will be optimistic relative to the walk-forward OOS metrics; the walk-forward metrics are the scientifically valid evaluation.

### GARCH persistence diagnostic

```
α + γ/2 + β < 1   →  covariance-stationary process
α + γ/2 + β = 1   →  integrated GARCH (I-GARCH): infinite variance persistence
α + γ/2 + β > 1   →  explosive process (ill-specified model)
```

Values above 0.97 are common for daily equity returns and do not indicate a problem; values ≥ 1.0 trigger a `warning()`.

---

## 13. Methodological Optimizations (v1 → v2)

This section documents the four changes introduced in v2, their academic rationale, and their practical implementation.

### [OPT-1] GRU replaces LSTM

**What changed:** `layer_lstm()` replaced with `layer_gru()` throughout.

**Why:**

LSTM has four gate matrices per layer (input, forget, output gates + cell-state candidate). GRU merges forget and input into a single update gate, eliminating the separate cell state. This gives GRU approximately 25% fewer parameters for the same hidden unit count.

For financial time series with low signal-to-noise ratio (~0.05–0.15) and limited per-fold samples (~200–250 sequences per fold), the capacity reduction acts as implicit regularisation. LSTM's additional parameters increase the risk of memorising noise patterns in individual regimes, which is especially damaging in walk-forward mode where each fold sees a different market environment.

**Parameter count comparison at 64/32 units, 2 features:**

| Model | Layer 1 params | Layer 2 params | Total |
|---|---|---|---|
| LSTM | 4×(64×2 + 64×64 + 64) = 17,664 | 4×(32×64 + 32×32 + 32) = 12,416 | 30,080 |
| GRU  | 3×(64×2 + 64×64 + 64) = 13,248 | 3×(32×64 + 32×32 + 32) = 9,312  | 22,560 |

GRU uses ~25% fewer parameters.

---

### [OPT-2] Variance feature transformation

**What changed:** `h_t` is passed through `APPLY_VAR_TRANSFORM()` before Min-Max scaling.

**Why:**

Raw conditional variance `h_t` from GJR-GARCH is heavy-tailed and right-skewed. During crisis periods (e.g., March 2020, October 2008), `h_t` spikes to 50–100× its long-run mean. Feeding raw `h_t` into Min-Max scaling creates two problems:

1. The scaler range is dominated by one or two extreme observations, compressing all non-crisis variation into the bottom 1–2% of `[0, 1]`.
2. The neural network sees near-identical inputs for all non-crisis days, destroying the discriminative power of the variance feature.

The `sqrt` transform (default) maps the tail exponent from ~4 (typical for squared daily returns) to ~2, similar to a normal distribution. The `log` transform maps multiplicative shocks to additive ones, analogous to the log-variance parameterisation in eGARCH.

**Skewness reduction example (typical SPX data):**

| Series | Skewness |
|---|---|
| Raw `h_t` | ~15–40 (extreme right skew) |
| `sqrt(h_t)` | ~3–5 (moderate right skew) |
| `log(h_t)` | ~0.5–1.5 (near-symmetric) |

---

### [OPT-3] Tightened GARCH solver tolerances

**What changed:** `solver.control = list(tol = 1e-12, delta = 1e-11)`.

**Why:**

The default `rugarch` convergence tolerance is `1e-8`. For GJR-GARCH models on daily equity returns with persistence near 0.97–0.99, the log-likelihood surface is extremely flat in the `β` direction — moving `β` by ±0.005 changes the log-likelihood by less than 0.001. The default tolerance allows the hybrid solver to declare convergence when still several gradient steps from the true MLE.

This systematically biases the estimated parameters: `β` is underestimated and `α` is overestimated, producing conditional variance paths that over-react to recent shocks and under-weight the long-run component. The tighter tolerances force the solver to continue until the gradient norm is genuinely negligible.

The `delta` parameter controls the finite-difference step used to compute the numerical Hessian for standard errors. The default is also loose; the tighter value produces more accurate standard errors, which matters if you use the GARCH fit for formal hypothesis testing.

---

### [OPT-4] Walk-forward validation

**What changed:** Static 80/20 chronological split replaced by rolling-window walk-forward protocol.

**Why:**

The static split evaluates performance on a single contiguous test period. If that period happens to be a calm (or crisis) regime unrepresentative of the full history, the RMSE metric is misleading. Walk-forward validation distributes the evaluation across multiple non-overlapping prediction windows, providing:

1. An empirical distribution of OOS error (mean, variance, worst-case fold).
2. Evidence about regime-specific performance — did the model fail during high-VIX periods?
3. Honest generalisation error, since each fold's GRU is tested on data that was never used in its training window.

The **regime-adaptive retraining** (fresh GRU per fold) ensures the model weights reflect the current volatility regime rather than accumulating gradients from remote historical periods. This is especially important for the SPX→ESG transfer: if the most recent fold learned weights calibrated to a post-COVID low-vol regime, those weights are more likely to produce sensible ESG synthetic VIX values than weights calibrated to a 2008 crisis regime.

---

## 14. Error Handling & Fallback Chains

### GARCH solver fallback

```
Primary:   hybrid [tol=1e-12, delta=1e-11]
           ↓ (on error)
Fallback 1: nlminb [tol=1e-12, delta=1e-11]
           ↓ (on error)
Fallback 2: solnp [default tolerances]
```

`solnp` is always available and does not use gradient-based convergence criteria, making it a reliable last resort at the cost of potentially looser parameter estimates.

### Model save fallback

```
Primary:   keras::save_model()     (keras3 >= 3.3)
           ↓ (on error: not exported)
Fallback 1: keras$saving$save_model()   (Python passthrough, always available)
           ↓ (on error)
Fallback 2: keras::save_model_weights_tf()  (weights only; requires architecture rebuild to reload)
```

### Output directory fallback

```
if /kaggle/working exists  → /kaggle/working
elif getOption("output_dir") set → that path
else → getwd()
```

### Walk-forward data guard

```r
if (pred_end_idx > n_total_seq) break
```

If the final fold's prediction window would exceed the available data, the loop exits cleanly rather than throwing an array out-of-bounds error. The metrics up to the last complete fold are still reported.

---

## 15. Extending the Pipeline

### Changing the variance transform

Set `VAR_TRANSFORM <- "log"` for maximum linearisation (recommended when the ESG index has extreme crisis behaviour). Set `"none"` to reproduce v1 behaviour for comparison.

### Switching to eGARCH

```r
GARCH_VARIANT <- "eGARCH"
```

eGARCH parameterises `log(h_t)` directly and therefore does not require the variance feature transform for gradient stability (the transform is still applied, but `log(log(h_t))` will be used if `VAR_TRANSFORM = "log"` — this is redundant and should be set to `"none"` when using eGARCH).

### Adding more features

To add a third feature (e.g., 5-day rolling realised variance `rv_t`):

```r
N_FEATURES <- 3L
scaler_rv   <- SCALE_FIT(rv_spx)
X_spx_scaled <- cbind(
  scaler_r_spx$transform(r_spx),
  scaler_fh_spx$transform(fh_spx),
  scaler_rv$transform(rv_spx)
)
```

Update `BUILD_GRU_MODEL()` input shape automatically via `n_features = dim(X_train)[3]`.

### Expanding vs. rolling window

To use an **expanding window** instead of rolling (train on all history up to fold k):

```r
train_start_idx <- 1L   # always start from the beginning
train_end_idx   <- TRAIN_DAYS + (k - 1L) * PRED_DAYS
```

Expanding windows accumulate more data per fold but may slow learning in non-stationary environments because early regime data dilutes recent signal.

### Fine-tuning (warm-start) across folds

Replace the per-fold `BUILD_GRU_MODEL()` call with:

```r
if (k == 1L) {
  gru_fold <- BUILD_GRU_MODEL()
} else {
  gru_fold <- keras::load_model(prev_ckpt_path)
  # Reduce LR by 10x for fine-tuning
  keras::compile(gru_fold,
    optimizer = keras::optimizer_adam(learning_rate = LR / 10),
    loss = "mean_squared_error",
    metrics = list("mean_absolute_error"))
}
```

### Reloading the saved model

```r
final_gru <- keras::load_model(
  file.path(OUTPUT_DIR, "gjr_garch_gru_model.keras")
)
# Re-run transfer inference:
synth_vix_scaled <- GRU_INFER(final_gru, X_esg_tensor)
```

---

## 16. Known Limitations

**Transfer validity depends on regime overlap.** If the ESG index was launched during a period not represented in the SPX training data, the GRU will extrapolate outside its training distribution. Always inspect plot `06_spx_garch_vol_transform.png` to confirm that the ESG conditional SD range is covered by the SPX training range.

**No option pricing.** The synthetic VIX produced is a neural-network-predicted *implied volatility index*, not a model-consistent implied volatility surface. It cannot be used directly for option pricing without additional calibration.

**Single-step ahead.** The sequence builder uses a many-to-one configuration with a single next-day target. Multi-step forecasting (e.g., a 5-day VIX term structure) would require a seq-to-seq architecture.

**Scale mismatch risk.** If the ESG index is systematically 2–3× more volatile than SPX (common for small-cap or sector ESG indices), the SPX VIX scaler's inverse transform will produce synthetic VIX values that are clipped at the SPX maximum. Inspect the output range and compare to `scaler_vix$max`.

**Memory in walk-forward loop.** `BUILD_SEQUENCES()` is called inside each fold iteration, which reconstructs the full sequence tensor `(N−T, T, 2)` every fold. For very large datasets this is redundant. Refactor by calling `BUILD_SEQUENCES()` once before the loop and slicing the result inside the loop.

**Stationarity assumption.** Both the GARCH model and the GRU are estimated under the implicit assumption that the data-generating process is stationary (or at least locally stationary within each fold window). Structural breaks (e.g., a permanent regime change in ESG index composition) violate this assumption.

---

## 17. References

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109, 1997–2028.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv:1406.1078*.

Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.

Heston, S. L., & Nandi, S. (2000). A closed-form GARCH option valuation model. *Review of Financial Studies*, 13(3), 585–625.

Ruan, Q., Zhang, S., & Luo, C. (2022). A universal volatility formation mechanism: Transfer learning for implied volatility forecasting. *International Review of Financial Analysis*, 84, 102385.

---

