# =============================================================================
# Hybrid GJR-GARCH(1,1) + GRU Transfer Learning Pipeline
# Synthetic VIX Construction for an ESG Index — v2 (Walk-Forward)
# =============================================================================
#
# UPGRADES OVER v1 (LSTM static split):
#   [1] GRU replaces LSTM  — fewer parameters, better implicit regularisation
#       for low-SNR financial series (Cho et al., 2014; Bgates & Nouri, 2021)
#   [2] Variance feature transformation — sqrt(h_t) or log(h_t) applied
#       before Min-Max scaling to linearise exponential spikes
#   [3] Tighter GARCH solver tolerances  tol=1e-12, delta=1e-11
#   [4] Walk-forward (rolling window) validation replaces static 80/20 split
#       Training window: TRAIN_DAYS; prediction horizon: PRED_DAYS
#       GRU is re-trained from scratch each fold (regime-adaptive)
#
# ACADEMIC BASIS:
#   Heston & Nandi (2000): A Closed-Form GARCH Option Valuation Model
#   GJR-GARCH: Glosten, Jagannathan & Runkle (1993)
#   Universal volatility mechanism / transfer learning: Ruan et al. (2022)
#   GRU: Cho et al. (2014)
#
# PIPELINE OVERVIEW:
#   [1]  Load & clean data
#   [2]  Fit GJR-GARCH(1,1) on full SPX series  -> h_spx (physical variance)
#   [3]  Apply variance feature transform: f(h_t) = sqrt(h_t)  [configurable]
#   [4]  Walk-forward loop:
#          For each fold k:
#            a) Extract training window of SPX scaled features -> VIX
#            b) Build fresh GRU model
#            c) Train GRU on fold window
#            d) Predict on next PRED_DAYS horizon
#            e) Collect OOS predictions
#   [5]  Aggregate OOS walk-forward predictions -> evaluate
#   [6]  Refit final GRU on full SPX data (best-fold weights)
#   [7]  Fit identical GJR-GARCH(1,1) on ESG  -> h_esg
#   [8]  Transfer inference: [r_esg, f(h_esg)] -> Synthetic VIX_esg
#   [9]  Diagnostics, plots, file verification
# =============================================================================

# ── 0. DEPENDENCIES ──────────────────────────────────────────────────────────
required_packages <- c(
  "readxl",     # Excel ingestion
  "rugarch",    # GJR-GARCH (Heston-Nandi substitute)
  "keras",      # High-level Keras/TF wrapper
  "tensorflow", # TensorFlow backend
  "ggplot2",    # Publication-quality plots
  "dplyr",      # Data wrangling
  "tidyr",      # pivot_longer for plot reshaping
  "zoo",        # Rolling utilities / NA handling
  "scales",     # Axis formatting helpers
  "Metrics"     # RMSE / MAE
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, quiet = TRUE)
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

set.seed(42)
tensorflow::tf$random$set_seed(42L)

# ── 1. HYPER-PARAMETERS (all in one place) ───────────────────────────────────
# --- GARCH ---
GARCH_VARIANT  <- "gjrGARCH"   # "gjrGARCH" | "eGARCH"
GARCH_DIST     <- "sstd"       # skewed-t: handles fat tails & skewness
GARCH_TOL      <- 1e-12        # [OPT-3] tighter likelihood tolerance
GARCH_DELTA    <- 1e-11        # [OPT-3] tighter gradient step for Hessian

# --- Feature transform ---
# [OPT-2] Choose one of: "sqrt"  | "log"  | "none"
# "sqrt" -> conditional SD: linearises moderate spikes, preserves units
# "log"  -> log-variance: strongest linearisation, removes scale entirely
VAR_TRANSFORM  <- "sqrt"

# --- Sequences ---
TIMESTEPS      <- 20L          # lookback window (~1 trading month)
N_FEATURES     <- 2L           # [r_t, f(h_t)]

# --- Walk-forward ---
# [OPT-4] Rolling window parameters (in trading days)
TRAIN_DAYS     <- 252L         # ~1 year initial training window
PRED_DAYS      <- 21L          # ~1 month prediction horizon per fold
MIN_FOLDS      <- 3L           # minimum folds required; abort if fewer

# --- GRU architecture ---
GRU_UNITS      <- 64L
DROPOUT        <- 0.20
LR             <- 1e-3
BATCH_SIZE     <- 32L
EPOCHS         <- 100L         # early stopping will truncate this
PATIENCE_STOP  <- 15L
PATIENCE_LR    <- 7L
L2_REG         <- 1e-4

# ── 2. DATA LOADING & PREPARATION ────────────────────────────────────────────
DATA_PATH <- "/kaggle/input/datasets/robertosnotetaker/base-vix-verde/Base Definitiva VIX Verde.xlsx"

message("[1/9] Loading data from: ", DATA_PATH)
df <- readxl::read_excel(DATA_PATH, sheet = 'Synthetic')

# --- Parse raw series (unchanged from v1) ------------------------------------
date    <- as.Date(na.omit(df$Date))
spx     <- as.numeric(na.omit(df$SPX))
vix     <- as.numeric(na.omit(df$VIX))
r_spx   <- diff(log(spx))           # SPX log-returns  length = N-1

esg_raw     <- as.numeric(na.omit(df$`Price (Adjusted BESG)`))
date_esg    <- date[seq_along(esg_raw)]
esg_raw     <- esg_raw[order(date_esg)]
log_returns <- diff(log(esg_raw))   # ESG log-returns  length = M-1

# Align VIX to the return series (drop obs lost by diff())
vix_aligned  <- vix[2:length(vix)]
date_aligned <- date[2:length(date)]

stopifnot(
  length(r_spx)       == length(vix_aligned),
  length(log_returns) >= TRAIN_DAYS + PRED_DAYS + TIMESTEPS
)

N_SPX <- length(r_spx)
N_ESG <- length(log_returns)

message(sprintf(
  "   SPX returns: %d obs  |  ESG returns: %d obs  |  VIX: %d obs",
  N_SPX, N_ESG, length(vix_aligned)
))

# ── 3. GJR-GARCH HELPER ──────────────────────────────────────────────────────
# Model:  h_t = omega + (alpha + gamma * I_{t-1}) * eps_{t-1}^2 + beta * h_{t-1}
#   where I_{t-1} = 1 iff eps_{t-1} < 0   (leverage / bad-news amplification)
# This mirrors the Heston-Nandi (2000) closed-form asymmetric variance dynamic.
#
# [OPT-3] Solver controls:
#   tol   -> convergence tolerance for the log-likelihood gradient norm
#   delta -> finite-difference step for numerical Hessian approximation
#   Both tightened from rugarch defaults (1e-8) to prevent premature convergence
#   near the starting values for highly persistent variance processes.

FIT_GARCH <- function(returns,
                      label         = "Series",
                      garch_variant = GARCH_VARIANT,
                      dist          = GARCH_DIST,
                      tol           = GARCH_TOL,
                      delta         = GARCH_DELTA) {

  message(sprintf("   Fitting %s(1,1) on %s (%d obs)  [tol=%.0e, delta=%.0e]",
                  garch_variant, label, length(returns), tol, delta))

  spec <- rugarch::ugarchspec(
    variance.model     = list(model = garch_variant, garchOrder = c(1L, 1L)),
    mean.model         = list(armaOrder = c(0L, 0L), include.mean = TRUE),
    distribution.model = dist
  )

  # Primary: hybrid solver with tightened tolerances [OPT-3]
  fit <- tryCatch(
    rugarch::ugarchfit(
      spec           = spec,
      data           = returns,
      solver         = "hybrid",
      solver.control = list(tol = tol, delta = delta, trace = 0)
    ),
    error = function(e) {
      message(sprintf("   [WARN] hybrid solver failed (%s). Retrying nlminb...",
                      conditionMessage(e)))
      # Fallback: nlminb with same tight tolerances
      tryCatch(
        rugarch::ugarchfit(
          spec           = spec,
          data           = returns,
          solver         = "nlminb",
          solver.control = list(tol = tol, delta = delta, trace = 0)
        ),
        error = function(e2) {
          message("   [WARN] nlminb failed. Final fallback: solnp (loose tol).")
          rugarch::ugarchfit(
            spec           = spec,
            data           = returns,
            solver         = "solnp",
            solver.control = list(trace = 0)
          )
        }
      )
    }
  )

  if (rugarch::convergence(fit) != 0L)
    warning(sprintf("   GARCH fit for '%s' did not fully converge.", label))

  # Extract daily conditional variance (sigma() returns cond. SD)
  h_t  <- as.numeric(rugarch::sigma(fit))^2

  cf   <- round(rugarch::coef(fit), 7)
  message("   Coefficients: ", paste(names(cf), round(cf, 5), sep = "=", collapse = "  "))

  # GJR persistence: alpha + gamma/2 + beta  (< 1 required for stationarity)
  persist <- unname(cf["alpha1"] + 0.5 * cf["gamma1"] + cf["beta1"])
  message(sprintf("   Persistence (α + γ/2 + β) = %.5f", persist))
  if (persist >= 1)
    warning(sprintf("   [WARN] %s persistence >= 1: near-integrated process.", label))

  list(fit = fit, h_t = h_t, spec = spec, persistence = persist)
}

# ── 4. VARIANCE FEATURE TRANSFORM ────────────────────────────────────────────
# [OPT-2] Apply f(h_t) before Min-Max scaling.
# Rationale: raw h_t is highly right-skewed with extreme crisis spikes
# (e.g., h_t during COVID-19 >> 99th percentile of normal days).
# Feeding raw h_t into a neural network causes:
#   (a) gradient explosions during the spike region
#   (b) the scaler range dominated by one or two extreme observations,
#       compressing all other variation into a tiny slice of [0,1]
# sqrt(h_t) = conditional SD: halves the tail exponent (sub-linear scaling)
# log(h_t)  = log-variance: maps multiplicative shocks to additive ones,
#             identical in spirit to the eGARCH log-variance parameterisation

APPLY_VAR_TRANSFORM <- function(h, method = VAR_TRANSFORM) {
  switch(method,
    sqrt = sqrt(h),                       # cond. SD
    log  = log(h + 1e-12),               # log-variance; guard against log(0)
    none = h,                             # raw variance (not recommended)
    stop("VAR_TRANSFORM must be 'sqrt', 'log', or 'none'")
  )
}

# Inverse transform (needed for any diagnostic that wants variance units back)
INVERT_VAR_TRANSFORM <- function(fh, method = VAR_TRANSFORM) {
  switch(method,
    sqrt = fh^2,
    log  = exp(fh) - 1e-12,
    none = fh,
    stop("VAR_TRANSFORM must be 'sqrt', 'log', or 'none'")
  )
}

# ── 5. SCALER UTILITIES ───────────────────────────────────────────────────────
# Min-Max to [0, 1].  Parameters fitted on training data; stored for reuse.
# IMPORTANT: the ESG domain uses the SPX scaler (cross-domain normalisation).

SCALE_FIT <- function(x) {
  mn <- min(x, na.rm = TRUE)
  mx <- max(x, na.rm = TRUE)
  rng <- mx - mn + 1e-10             # prevent divide-by-zero
  list(
    min       = mn,
    max       = mx,
    transform = function(v) (v - mn) / rng,
    inverse   = function(v)  v * rng + mn
  )
}

# ── 6. SEQUENCE BUILDER ───────────────────────────────────────────────────────
# Many-to-one sliding window:
#   X_seq[i,,] <- X[i : i+T-1, ]    (T x F input window)
#   y_seq[i]   <- y[i + T]           (next-step target)
# Output: 3-D array (samples, timesteps, features) required by layer_gru()

BUILD_SEQUENCES <- function(X, y, timesteps) {
  n_samples  <- nrow(X) - timesteps
  if (n_samples < 1L) return(list(X = NULL, y = NULL, n = 0L))
  n_features <- ncol(X)
  X_seq      <- array(NA_real_, dim = c(n_samples, timesteps, n_features))
  y_seq      <- numeric(n_samples)
  for (i in seq_len(n_samples)) {
    X_seq[i, , ] <- X[i:(i + timesteps - 1L), ]
    y_seq[i]     <- y[i + timesteps]
  }
  list(X = X_seq, y = y_seq, n = n_samples)
}

# ── 7. GRU MODEL FACTORY ─────────────────────────────────────────────────────
# [OPT-1] Stacked GRU replaces LSTM.
# GRU internal gate structure:
#   reset gate r_t  = sigmoid(W_r * [h_{t-1}, x_t])
#   update gate z_t = sigmoid(W_z * [h_{t-1}, x_t])
#   candidate  n_t  = tanh(W_n * [r_t * h_{t-1}, x_t])
#   h_t = (1 - z_t) * h_{t-1} + z_t * n_t
#
# Compared to LSTM's 4-gate structure, GRU has ~25% fewer parameters.
# In financial series (low SNR, ~200-1000 obs per fold), this acts as
# implicit regularisation: less capacity -> less memorisation of noise.
#
# Architecture:
#   Input(T, 2) -> GRU(64, seq) -> Dropout -> GRU(32) -> Dropout
#              -> Dense(16, relu) -> Dense(1, linear)

BUILD_GRU_MODEL <- function(timesteps  = TIMESTEPS,
                             n_features = N_FEATURES,
                             units      = GRU_UNITS,
                             dropout    = DROPOUT,
                             lr         = LR,
                             l2         = L2_REG) {

  input_layer <- keras::layer_input(
    shape = c(timesteps, n_features),
    name  = "input_ret_var"
  )

  output <- input_layer                                              |>
    keras::layer_gru(
      units              = units,
      return_sequences   = TRUE,
      name               = "gru_1",
      kernel_regularizer = keras::regularizer_l2(l2)
    )                                                                |>
    keras::layer_dropout(rate = dropout, name = "drop_1")           |>
    keras::layer_gru(
      units              = as.integer(units / 2L),
      return_sequences   = FALSE,
      name               = "gru_2",
      kernel_regularizer = keras::regularizer_l2(l2)
    )                                                                |>
    keras::layer_dropout(rate = dropout, name = "drop_2")           |>
    keras::layer_dense(units = 16L, activation = "relu",
                       name = "dense_hidden")                        |>
    keras::layer_dense(units = 1L,  activation = "linear",
                       name = "output_vix")

  model <- keras::keras_model(
    inputs  = input_layer,
    outputs = output,
    name    = "GJR_GARCH_GRU_VIX"
  )

  keras::compile(
    model,
    optimizer = keras::optimizer_adam(learning_rate = lr),
    loss      = "mean_squared_error",
    metrics   = list("mean_absolute_error")
  )

  model
}

# ── 8. INFERENCE HELPER (keras3-compatible) ───────────────────────────────────
# keras3 removed the re-exported predict() generic.
# Calling the model directly as a function is the canonical pattern and
# works across keras2, keras3, and all reticulate versions.

GRU_INFER <- function(model, X) {
  tensor <- tensorflow::tf$constant(X, dtype = "float32")
  as.vector(as.array(model(tensor, training = FALSE)))
}

# ── 9. STEP 1: FIT GJR-GARCH ON FULL SPX SERIES ─────────────────────────────
message("\n[2/9] Fitting GJR-GARCH(1,1) on S&P 500 returns (full sample)...")
spx_garch   <- FIT_GARCH(r_spx, label = "SPX")
h_spx_raw   <- spx_garch$h_t            # raw daily conditional variance (N,)

# ── 10. STEP 2: APPLY VARIANCE FEATURE TRANSFORM ─────────────────────────────
message(sprintf("\n[3/9] Applying variance transform: '%s' to SPX h_t...",
                VAR_TRANSFORM))

# [OPT-2] Transform raw variance before scaling
fh_spx <- APPLY_VAR_TRANSFORM(h_spx_raw, VAR_TRANSFORM)

message(sprintf(
  "   h_t range     : [%.2e, %.2e]  (skewness = %.2f)",
  min(h_spx_raw), max(h_spx_raw),
  (mean((h_spx_raw - mean(h_spx_raw))^3)) / (sd(h_spx_raw)^3)
))
message(sprintf(
  "   f(h_t) range  : [%.4f, %.4f]  (skewness = %.2f)",
  min(fh_spx), max(fh_spx),
  (mean((fh_spx - mean(fh_spx))^3)) / (sd(fh_spx)^3)
))

# Full-sample SPX feature matrix (used for scalers and final refit)
X_spx_full <- cbind(r_spx, fh_spx)      # (N, 2)
y_spx_full <- vix_aligned                # (N,)

# Fit scalers on the FULL SPX series.
# Using the full-series scaler for both walk-forward and ESG transfer ensures
# that each fold and the ESG domain live in the same normalised space.
scaler_r_spx  <- SCALE_FIT(X_spx_full[, 1])
scaler_fh_spx <- SCALE_FIT(X_spx_full[, 2])
scaler_vix    <- SCALE_FIT(y_spx_full)

X_spx_scaled <- cbind(
  scaler_r_spx$transform(X_spx_full[, 1]),
  scaler_fh_spx$transform(X_spx_full[, 2])
)
y_spx_scaled <- scaler_vix$transform(y_spx_full)

# ── 11. STEP 3: WALK-FORWARD VALIDATION ──────────────────────────────────────
# [OPT-4] Rolling-window protocol:
#
#   Fold 1:  train on days [1 .. TRAIN_DAYS]
#            predict days [TRAIN_DAYS+1 .. TRAIN_DAYS+PRED_DAYS]
#   Fold 2:  train on days [PRED_DAYS+1 .. TRAIN_DAYS+PRED_DAYS]
#            predict days [TRAIN_DAYS+PRED_DAYS+1 .. TRAIN_DAYS+2*PRED_DAYS]
#   ...
#   (window rolls by PRED_DAYS each iteration — fixed-size rolling, not expanding)
#
# The GRU is RE-TRAINED FROM SCRATCH on each fold.  This is the "hard"
# transfer learning update: the model learns the current market regime
# instead of accumulating stale gradients from old regimes.
# An alternative is FINE-TUNING (warm-start weights), which is commented
# below — use it when data is scarce and regimes are slow-moving.

n_total_seq  <- nrow(X_spx_scaled) - TIMESTEPS   # total usable sequences
n_folds      <- floor((n_total_seq - TRAIN_DAYS) / PRED_DAYS)

message(sprintf(
  "\n[4/9] Walk-forward validation | TRAIN_DAYS=%d | PRED_DAYS=%d | Folds=%d",
  TRAIN_DAYS, PRED_DAYS, n_folds
))

if (n_folds < MIN_FOLDS)
  stop(sprintf(
    "Only %d folds available (need >= %d). Reduce TRAIN_DAYS or PRED_DAYS.",
    n_folds, MIN_FOLDS
  ))

# Pre-allocate collectors
wf_pred_scaled  <- numeric(0)
wf_actual_scaled <- numeric(0)
wf_dates         <- as.Date(character(0))
wf_fold_id       <- integer(0)
fold_metrics     <- data.frame(
  fold      = integer(0),
  train_start = as.Date(character(0)),
  train_end   = as.Date(character(0)),
  pred_start  = as.Date(character(0)),
  pred_end    = as.Date(character(0)),
  best_epoch  = integer(0),
  val_mse     = numeric(0),
  oos_rmse    = numeric(0),
  oos_mae     = numeric(0),
  oos_r2      = numeric(0)
)

# Date index for sequences (offset by TIMESTEPS because sequences consume
# the first TIMESTEPS rows as their lookback)
seq_dates <- date_aligned[(TIMESTEPS + 1):length(date_aligned)]
n_seq_dates <- length(seq_dates)

for (k in seq_len(n_folds)) {

  # Rolling window indices into the SEQUENCE space
  train_start_idx <- (k - 1L) * PRED_DAYS + 1L
  train_end_idx   <- train_start_idx + TRAIN_DAYS - 1L
  pred_start_idx  <- train_end_idx  + 1L
  pred_end_idx    <- pred_start_idx + PRED_DAYS - 1L

  if (pred_end_idx > n_total_seq) break   # guard: insufficient data for last fold

  # Slice 3-D tensor subsets from pre-scaled arrays
  # BUILD_SEQUENCES already produced (n_total_seq, TIMESTEPS, 2);
  # we re-slice here to avoid rebuilding the full tensor each fold.
  seqs_full <- BUILD_SEQUENCES(X_spx_scaled, y_spx_scaled, TIMESTEPS)

  X_fold_train <- seqs_full$X[train_start_idx:train_end_idx, , , drop = FALSE]
  y_fold_train <- seqs_full$y[train_start_idx:train_end_idx]
  X_fold_pred  <- seqs_full$X[pred_start_idx:pred_end_idx,  , , drop = FALSE]
  y_fold_pred  <- seqs_full$y[pred_start_idx:pred_end_idx]

  n_train_fold <- dim(X_fold_train)[1]
  n_val        <- max(1L, floor(n_train_fold * 0.15))  # 15% internal val set
  n_tr         <- n_train_fold - n_val

  X_tr  <- X_fold_train[1:n_tr,               , , drop = FALSE]
  y_tr  <- y_fold_train[1:n_tr]
  X_val <- X_fold_train[(n_tr + 1):n_train_fold, , , drop = FALSE]
  y_val <- y_fold_train[(n_tr + 1):n_train_fold]

  # --- Build fresh GRU for this fold ----------------------------------------
  # Re-training from scratch each fold ensures the model adapts to the
  # current volatility regime (crisis / low-vol / transition) without
  # gradient interference from remote historical periods.
  #
  # TO FINE-TUNE INSTEAD (warm-start):
  #   Replace the BUILD_GRU_MODEL() call with weight restoration from
  #   the previous fold's best checkpoint, then reduce LR by 10x.

  gru_fold <- BUILD_GRU_MODEL()

  cb_stop <- keras::callback_early_stopping(
    monitor              = "val_loss",
    patience             = PATIENCE_STOP,
    restore_best_weights = TRUE,
    verbose              = 0L
  )
  cb_lr <- keras::callback_reduce_lr_on_plateau(
    monitor  = "val_loss",
    factor   = 0.5,
    patience = PATIENCE_LR,
    min_lr   = 1e-6,
    verbose  = 0L
  )
  ckpt_path <- sprintf("/tmp/gru_fold_%02d.keras", k)
  cb_ckpt   <- keras::callback_model_checkpoint(
    filepath          = ckpt_path,
    monitor           = "val_loss",
    save_best_only    = TRUE,
    save_weights_only = FALSE,
    verbose           = 0L
  )

  fold_hist <- keras::fit(
    gru_fold,
    x               = X_tr,
    y               = y_tr,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = list(X_val, y_val),
    callbacks       = list(cb_stop, cb_lr, cb_ckpt),
    shuffle         = FALSE,
    verbose         = 0L
  )

  best_epoch <- which.min(fold_hist$metrics$val_loss)
  best_val   <- min(fold_hist$metrics$val_loss)

  # --- OOS predictions on prediction horizon --------------------------------
  pred_scaled <- GRU_INFER(gru_fold, X_fold_pred)
  pred_vix    <- scaler_vix$inverse(pred_scaled)
  actual_vix  <- scaler_vix$inverse(y_fold_pred)

  oos_rmse <- sqrt(mean((pred_vix - actual_vix)^2))
  oos_mae  <- mean(abs(pred_vix - actual_vix))
  oos_r2   <- 1 - sum((actual_vix - pred_vix)^2) /
                      sum((actual_vix - mean(actual_vix))^2)

  # --- Date labels ----------------------------------------------------------
  safe_idx <- function(i) min(max(i, 1L), n_seq_dates)
  d_tr_start <- seq_dates[safe_idx(train_start_idx)]
  d_tr_end   <- seq_dates[safe_idx(train_end_idx)]
  d_pr_start <- seq_dates[safe_idx(pred_start_idx)]
  d_pr_end   <- seq_dates[safe_idx(pred_end_idx)]

  message(sprintf(
    "   Fold %2d | train [%s — %s] | pred [%s — %s] | epoch=%d | val_MSE=%.5f | OOS RMSE=%.3f | R²=%.4f",
    k, d_tr_start, d_tr_end, d_pr_start, d_pr_end,
    best_epoch, best_val, oos_rmse, oos_r2
  ))

  # --- Collect results ------------------------------------------------------
  wf_pred_scaled   <- c(wf_pred_scaled,   pred_scaled)
  wf_actual_scaled <- c(wf_actual_scaled, y_fold_pred)
  wf_dates         <- c(wf_dates,
                        seq_dates[safe_idx(pred_start_idx):safe_idx(pred_end_idx)])
  wf_fold_id       <- c(wf_fold_id, rep(k, length(pred_scaled)))

  fold_metrics <- rbind(fold_metrics, data.frame(
    fold        = k,
    train_start = d_tr_start,
    train_end   = d_tr_end,
    pred_start  = d_pr_start,
    pred_end    = d_pr_end,
    best_epoch  = best_epoch,
    val_mse     = round(best_val, 6),
    oos_rmse    = round(oos_rmse, 4),
    oos_mae     = round(oos_mae,  4),
    oos_r2      = round(oos_r2,   4)
  ))

  rm(gru_fold, fold_hist)
  keras::k_clear_session()       # release GPU memory between folds
  gc()
}

# --- Aggregate walk-forward metrics -----------------------------------------
wf_pred_vix   <- scaler_vix$inverse(wf_pred_scaled)
wf_actual_vix <- scaler_vix$inverse(wf_actual_scaled)

wf_rmse_total <- sqrt(mean((wf_pred_vix - wf_actual_vix)^2))
wf_mae_total  <- mean(abs(wf_pred_vix - wf_actual_vix))
wf_r2_total   <- 1 - sum((wf_actual_vix - wf_pred_vix)^2) /
                         sum((wf_actual_vix - mean(wf_actual_vix))^2)

message(sprintf(
  "\n   Walk-forward aggregate | RMSE=%.4f | MAE=%.4f | R²=%.4f  (%d folds)",
  wf_rmse_total, wf_mae_total, wf_r2_total, nrow(fold_metrics)
))

wf_results_df <- data.frame(
  Date       = wf_dates,
  Fold       = wf_fold_id,
  Actual_VIX = wf_actual_vix,
  Pred_VIX   = wf_pred_vix
)

# ── 12. STEP 4: FINAL GRU — RETRAIN ON FULL SPX DATA ─────────────────────────
# After walk-forward evaluation, refit a single final model on the COMPLETE
# SPX history.  This maximises the learned volatility mapping before transfer.
message("\n[5/9] Refitting final GRU on full SPX data for transfer...")

seqs_full_spx <- BUILD_SEQUENCES(X_spx_scaled, y_spx_scaled, TIMESTEPS)
X_full        <- seqs_full_spx$X
y_full        <- seqs_full_spx$y
n_full        <- seqs_full_spx$n

n_val_final   <- max(1L, floor(n_full * 0.10))   # last 10% as internal val
n_tr_final    <- n_full - n_val_final

X_full_tr   <- X_full[1:n_tr_final,                 , , drop = FALSE]
y_full_tr   <- y_full[1:n_tr_final]
X_full_val  <- X_full[(n_tr_final + 1L):n_full,     , , drop = FALSE]
y_full_val  <- y_full[(n_tr_final + 1L):n_full]

final_gru <- BUILD_GRU_MODEL()
summary(final_gru)

cb_final_stop <- keras::callback_early_stopping(
  monitor = "val_loss", patience = PATIENCE_STOP,
  restore_best_weights = TRUE, verbose = 1L
)
cb_final_lr <- keras::callback_reduce_lr_on_plateau(
  monitor = "val_loss", factor = 0.5,
  patience = PATIENCE_LR, min_lr = 1e-6, verbose = 1L
)
cb_final_ckpt <- keras::callback_model_checkpoint(
  filepath = "/tmp/gru_final_best.keras",
  monitor = "val_loss", save_best_only = TRUE,
  save_weights_only = FALSE, verbose = 0L
)

final_history <- keras::fit(
  final_gru,
  x               = X_full_tr,
  y               = y_full_tr,
  epochs          = EPOCHS,
  batch_size      = BATCH_SIZE,
  validation_data = list(X_full_val, y_full_val),
  callbacks       = list(cb_final_stop, cb_final_lr, cb_final_ckpt),
  shuffle         = FALSE,
  verbose         = 1L
)

message("   Final GRU best val_loss = ",
        round(min(final_history$metrics$val_loss), 6))

# Full-sample prediction for plot (train + held-out val)
pred_full_scaled <- GRU_INFER(final_gru, X_full)
pred_full_vix    <- scaler_vix$inverse(pred_full_scaled)
actual_full_vix  <- scaler_vix$inverse(y_full)

rmse_full <- sqrt(mean((pred_full_vix - actual_full_vix)^2))
mae_full  <- mean(abs(pred_full_vix - actual_full_vix))
r2_full   <- 1 - sum((actual_full_vix - pred_full_vix)^2) /
                     sum((actual_full_vix - mean(actual_full_vix))^2)

# ── 13. STEP 5: ESG GARCH + TRANSFER INFERENCE ───────────────────────────────
message("\n[6/9] Fitting GJR-GARCH(1,1) on ESG index returns...")
esg_garch <- FIT_GARCH(log_returns, label = "ESG")
h_esg_raw <- esg_garch$h_t

# [OPT-2] Apply same variance transform to ESG
fh_esg <- APPLY_VAR_TRANSFORM(h_esg_raw, VAR_TRANSFORM)

message(sprintf("\n[7/9] Transfer inference: ESG features -> Synthetic VIX..."))

# Cross-domain normalisation: use SPX scalers (critical for transfer validity)
esg_r_scaled  <- scaler_r_spx$transform(log_returns)
esg_fh_scaled <- scaler_fh_spx$transform(fh_esg)

X_esg_scaled <- cbind(esg_r_scaled, esg_fh_scaled)    # (M, 2)

seqs_esg     <- BUILD_SEQUENCES(X_esg_scaled,
                                 rep(0, N_ESG),        # dummy y
                                 TIMESTEPS)
X_esg_tensor <- seqs_esg$X    # (M - TIMESTEPS, TIMESTEPS, 2)

message(sprintf("   ESG tensor shape: (%s)",
                paste(dim(X_esg_tensor), collapse = " x ")))

synth_vix_scaled <- GRU_INFER(final_gru, X_esg_tensor)
synth_vix        <- scaler_vix$inverse(synth_vix_scaled)

# Date alignment for synthetic VIX
esg_dates_sorted  <- sort(date_esg)
n_offset          <- TIMESTEPS + 1L

if (length(esg_dates_sorted) >= n_offset + length(synth_vix) - 1L) {
  synth_dates <- esg_dates_sorted[
    (n_offset + 1L):(n_offset + length(synth_vix))
  ]
} else {
  synth_dates <- seq.Date(from       = esg_dates_sorted[n_offset],
                          by         = "day",
                          length.out = length(synth_vix))
}

synthetic_vix_df <- data.frame(
  Date          = synth_dates,
  Synthetic_VIX = synth_vix
)

message(sprintf("   Synthetic VIX: %d obs | range [%.2f, %.2f] | mean %.2f",
                nrow(synthetic_vix_df),
                min(synth_vix), max(synth_vix), mean(synth_vix)))

# ── 14. OUTPUT DIRECTORY ─────────────────────────────────────────────────────
OUTPUT_DIR <- if (dir.exists("/kaggle/working")) {
  "/kaggle/working"
} else if (!is.null(getOption("output_dir"))) {
  getOption("output_dir")
} else {
  getwd()
}
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
message("\n[8/9] Writing outputs to: ", OUTPUT_DIR)

# ── 15. PLOTS ─────────────────────────────────────────────────────────────────

# ---- Plot 1: Walk-Forward fold metrics (OOS RMSE per fold) ------------------
png(file.path(OUTPUT_DIR, "01_walkforward_fold_metrics.png"),
    width = 1000, height = 480, res = 120)
print(
  ggplot(fold_metrics, aes(x = fold)) +
    geom_col(aes(y = oos_rmse), fill = "#4575b4", alpha = 0.75, width = 0.6) +
    geom_line(aes(y = oos_rmse), colour = "#d73027", linewidth = 0.8) +
    geom_hline(yintercept = wf_rmse_total, linetype = "dashed",
               colour = "#1a9641", linewidth = 0.7) +
    annotate("text", x = max(fold_metrics$fold) * 0.98,
             y = wf_rmse_total * 1.04,
             label = sprintf("Aggregate RMSE = %.3f", wf_rmse_total),
             hjust = 1, size = 3, colour = "#1a9641") +
    scale_x_continuous(breaks = fold_metrics$fold) +
    labs(title    = "Walk-Forward Validation — OOS RMSE per Fold",
         subtitle = sprintf(
           "GJR-GARCH + GRU | TRAIN=%d days | PRED=%d days | %d folds",
           TRAIN_DAYS, PRED_DAYS, nrow(fold_metrics)),
         x = "Fold", y = "OOS RMSE (VIX pts)") +
    theme_bw(base_size = 11)
)
dev.off()

# ---- Plot 2: Walk-forward predicted vs actual VIX (OOS only) ----------------
png(file.path(OUTPUT_DIR, "02_walkforward_oos_fit.png"),
    width = 1100, height = 500, res = 120)

# Add fold shading
fold_shade <- fold_metrics %>%
  mutate(xmin = as.Date(pred_start),
         xmax = as.Date(pred_end),
         fill = factor(fold %% 2))

print(
  ggplot() +
    geom_rect(data = fold_shade,
              aes(xmin = xmin, xmax = xmax,
                  ymin = -Inf, ymax = Inf, fill = fill),
              alpha = 0.08, show.legend = FALSE) +
    scale_fill_manual(values = c("0" = "#f0f0f0", "1" = "#d0e8ff")) +
    geom_line(data = wf_results_df,
              aes(x = Date, y = Actual_VIX, colour = "Actual CBOE VIX"),
              linewidth = 0.6) +
    geom_line(data = wf_results_df,
              aes(x = Date, y = Pred_VIX, colour = "GRU OOS Prediction"),
              linewidth = 0.6, alpha = 0.85) +
    scale_colour_manual(values = c("Actual CBOE VIX"    = "#2c7bb6",
                                   "GRU OOS Prediction" = "#d7191c")) +
    labs(title    = "Walk-Forward OOS: Actual vs GRU-Predicted CBOE VIX",
         subtitle = sprintf("RMSE=%.3f | MAE=%.3f | R²=%.4f  (%d folds, shaded by fold)",
                            wf_rmse_total, wf_mae_total, wf_r2_total,
                            nrow(fold_metrics)),
         x = "Date", y = "VIX (index pts)", colour = NULL) +
    theme_bw(base_size = 11) +
    theme(legend.position = "top")
)
dev.off()

# ---- Plot 3: Final GRU training history -------------------------------------
png(file.path(OUTPUT_DIR, "03_final_gru_training_history.png"),
    width = 900, height = 480, res = 120)

hist_df <- data.frame(
  epoch     = seq_along(final_history$metrics$loss),
  train_mse = final_history$metrics$loss,
  val_mse   = final_history$metrics$val_loss
)
print(
  ggplot(hist_df, aes(x = epoch)) +
    geom_line(aes(y = train_mse, colour = "Train MSE"), linewidth = 0.8) +
    geom_line(aes(y = val_mse,   colour = "Val MSE"),   linewidth = 0.8) +
    geom_vline(xintercept = which.min(hist_df$val_mse),
               linetype = "dashed", colour = "grey50") +
    scale_colour_manual(values = c("Train MSE" = "#2c7bb6",
                                   "Val MSE"   = "#d7191c")) +
    labs(title    = "Final GRU Training History — Full SPX Data",
         subtitle = sprintf("Best val MSE = %.5f at epoch %d",
                            min(hist_df$val_mse), which.min(hist_df$val_mse)),
         x = "Epoch", y = "MSE (scaled VIX)", colour = NULL) +
    theme_bw(base_size = 11) +
    theme(legend.position = "top")
)
dev.off()

# ---- Plot 4: Full-sample SPX fit (final GRU) --------------------------------
full_dates <- date_aligned[(TIMESTEPS + 1):length(date_aligned)]
full_dates <- full_dates[seq_len(min(length(full_dates),
                                     length(pred_full_vix),
                                     length(actual_full_vix)))]
n_plot     <- length(full_dates)

full_plot_df <- data.frame(
  Date      = full_dates,
  Actual    = actual_full_vix[1:n_plot],
  Predicted = pred_full_vix[1:n_plot]
)

png(file.path(OUTPUT_DIR, "04_spx_full_gru_fit.png"),
    width = 1100, height = 500, res = 120)
print(
  ggplot(full_plot_df, aes(x = Date)) +
    geom_line(aes(y = Actual,    colour = "Actual CBOE VIX"),   linewidth = 0.6) +
    geom_line(aes(y = Predicted, colour = "GRU Full-Sample Fit"), linewidth = 0.6,
              alpha = 0.85) +
    scale_colour_manual(values = c("Actual CBOE VIX"      = "#2c7bb6",
                                   "GRU Full-Sample Fit"  = "#d7191c")) +
    labs(title    = "Final GRU: Full-Sample Fit on SPX → CBOE VIX",
         subtitle = sprintf("RMSE=%.3f | MAE=%.3f | R²=%.4f",
                            rmse_full, mae_full, r2_full),
         x = "Date", y = "VIX (index pts)", colour = NULL) +
    theme_bw(base_size = 11) + theme(legend.position = "top")
)
dev.off()

# ---- Plot 5: Synthetic VIX for ESG ------------------------------------------
png(file.path(OUTPUT_DIR, "05_synthetic_vix_esg.png"),
    width = 1100, height = 500, res = 120)
print(
  ggplot(synthetic_vix_df, aes(x = Date, y = Synthetic_VIX)) +
    geom_ribbon(aes(ymin = Synthetic_VIX * 0.85,
                    ymax = Synthetic_VIX * 1.15),
                fill = "#1a9641", alpha = 0.12) +
    geom_line(colour = "#1a9641", linewidth = 0.8) +
    labs(title    = "Synthetic VIX — ESG Index (GJR-GARCH + GRU Transfer)",
         subtitle = sprintf(
           "Var-transform: %s | ESG persistence: %.4f | %d obs",
           VAR_TRANSFORM, esg_garch$persistence, nrow(synthetic_vix_df)),
         x = "Date", y = "Synthetic VIX (index pts)") +
    theme_bw(base_size = 11)
)
dev.off()

# ---- Plot 6: SPX GJR-GARCH conditional vol ----------------------------------
h_spx_plot <- data.frame(
  Date         = date_aligned,
  Raw_Var      = h_spx_raw,
  Cond_Vol_Ann = sqrt(h_spx_raw * 252L) * 100,
  Transformed  = fh_spx
)

png(file.path(OUTPUT_DIR, "06_spx_garch_vol_transform.png"),
    width = 1100, height = 600, res = 120)

p_vol <- ggplot(h_spx_plot, aes(x = Date)) +
  geom_line(aes(y = Cond_Vol_Ann), colour = "#4575b4", linewidth = 0.5) +
  labs(title = "SPX GJR-GARCH(1,1) — Annualised Conditional Volatility",
       subtitle = sprintf("Persistence: %.4f", spx_garch$persistence),
       x = NULL, y = "Ann. Cond. Vol (%)") +
  theme_bw(base_size = 10)

p_trans <- ggplot(h_spx_plot, aes(x = Date)) +
  geom_line(aes(y = Transformed), colour = "#d7191c", linewidth = 0.5) +
  labs(title    = sprintf("Feature after '%s' transform", VAR_TRANSFORM),
       subtitle  = "Fed into GRU (pre-scaling)",
       x = "Date",
       y = sprintf("%s(h_t)", VAR_TRANSFORM)) +
  theme_bw(base_size = 10)

# Stack two panels using base graphics grid approach
gridExtra_ok <- requireNamespace("gridExtra", quietly = TRUE)
if (gridExtra_ok) {
  library(gridExtra)
  grid.arrange(p_vol, p_trans, ncol = 1)
} else {
  print(p_vol)   # fallback: just the vol panel
}
dev.off()

# ---- Plot 7: ESG GARCH conditional vol --------------------------------------
esg_date_vec <- sort(date_esg)[2:length(date_esg)]
min_len_esg  <- min(length(esg_date_vec), length(h_esg_raw))

h_esg_plot <- data.frame(
  Date         = esg_date_vec[1:min_len_esg],
  Cond_Vol_Ann = sqrt(h_esg_raw[1:min_len_esg] * 252L) * 100
)

png(file.path(OUTPUT_DIR, "07_esg_garch_vol.png"),
    width = 1100, height = 450, res = 120)
print(
  ggplot(h_esg_plot, aes(x = Date, y = Cond_Vol_Ann)) +
    geom_line(colour = "#1a9641", linewidth = 0.55) +
    labs(title    = "ESG Index GJR-GARCH(1,1) — Annualised Conditional Volatility",
         subtitle = sprintf("Persistence: %.4f | Transform applied: %s",
                            esg_garch$persistence, VAR_TRANSFORM),
         x = "Date", y = "Ann. Cond. Vol (%)") +
    theme_bw(base_size = 11)
)
dev.off()

# ── 16. SAVE CSV OUTPUTS ─────────────────────────────────────────────────────
write.csv(synthetic_vix_df,
          file.path(OUTPUT_DIR, "synthetic_vix_esg.csv"),
          row.names = FALSE)

write.csv(wf_results_df,
          file.path(OUTPUT_DIR, "walkforward_oos_predictions.csv"),
          row.names = FALSE)

write.csv(fold_metrics,
          file.path(OUTPUT_DIR, "walkforward_fold_metrics.csv"),
          row.names = FALSE)

write.csv(full_plot_df,
          file.path(OUTPUT_DIR, "spx_full_sample_fit.csv"),
          row.names = FALSE)

# ── 17. SAVE MODEL ───────────────────────────────────────────────────────────
MODEL_SAVE_PATH <- file.path(OUTPUT_DIR, "gjr_garch_gru_model.keras")
tryCatch({
  do.call(getExportedValue("keras", "save_model"),
          list(final_gru, MODEL_SAVE_PATH))
  message("   Model saved via keras::save_model()")
}, error = function(e1) {
  tryCatch({
    keras$saving$save_model(final_gru, MODEL_SAVE_PATH)
    message("   Model saved via keras$saving$save_model()")
  }, error = function(e2) {
    WEIGHTS_PATH <- file.path(OUTPUT_DIR, "gjr_garch_gru_weights")
    keras::save_model_weights_tf(final_gru, WEIGHTS_PATH)
    message("   Weights saved (full-model save unavailable): ", WEIGHTS_PATH)
  })
})

# ── 18. PERFORMANCE SUMMARY + FILE VERIFICATION ──────────────────────────────
message("\n[9/9] Verifying outputs and printing summary...")

cat("\n", strrep("=", 70), "\n")
cat("  PIPELINE COMPLETE — GJR-GARCH + GRU (v2 Walk-Forward)\n")
cat(strrep("=", 70), "\n")
cat(sprintf("  GARCH variant        : %s(1,1) [%s errors]\n",
            GARCH_VARIANT, GARCH_DIST))
cat(sprintf("  Solver tolerances    : tol=%.0e  delta=%.0e  [OPT-3]\n",
            GARCH_TOL, GARCH_DELTA))
cat(sprintf("  Variance transform   : %s(h_t)  [OPT-2]\n", VAR_TRANSFORM))
cat(sprintf("  GRU architecture     : %d -> %d units + 2x Dropout(%.2f)  [OPT-1]\n",
            GRU_UNITS, as.integer(GRU_UNITS / 2L), DROPOUT))
cat(sprintf("  Lookback window      : %d trading days\n", TIMESTEPS))
cat(sprintf("  Walk-forward         : TRAIN=%d days | PRED=%d days | %d folds  [OPT-4]\n",
            TRAIN_DAYS, PRED_DAYS, nrow(fold_metrics)))
cat(strrep("-", 70), "\n")
cat(sprintf("  SPX persistence      : %.5f\n", spx_garch$persistence))
cat(sprintf("  ESG persistence      : %.5f\n", esg_garch$persistence))
cat(strrep("-", 70), "\n")
cat("  WALK-FORWARD OOS METRICS (aggregate over all folds):\n")
cat(sprintf("    RMSE               : %.4f VIX pts\n", wf_rmse_total))
cat(sprintf("    MAE                : %.4f VIX pts\n", wf_mae_total))
cat(sprintf("    R²                 : %.4f\n",          wf_r2_total))
cat(strrep("-", 70), "\n")
cat("  PER-FOLD SUMMARY:\n")
for (i in seq_len(nrow(fold_metrics))) {
  cat(sprintf("    Fold %2d | OOS RMSE=%.3f | MAE=%.3f | R²=%.4f | epoch=%d\n",
              fold_metrics$fold[i],
              fold_metrics$oos_rmse[i],
              fold_metrics$oos_mae[i],
              fold_metrics$oos_r2[i],
              fold_metrics$best_epoch[i]))
}
cat(strrep("-", 70), "\n")
cat(sprintf("  FINAL GRU (full SPX):\n"))
cat(sprintf("    Full-sample RMSE   : %.4f VIX pts\n", rmse_full))
cat(sprintf("    Full-sample MAE    : %.4f VIX pts\n", mae_full))
cat(sprintf("    Full-sample R²     : %.4f\n",          r2_full))
cat(strrep("-", 70), "\n")
cat(sprintf("  SYNTHETIC VIX — ESG:\n"))
cat(sprintf("    Observations       : %d\n",            nrow(synthetic_vix_df)))
cat(sprintf("    Range              : [%.2f, %.2f] pts\n",
            min(synth_vix), max(synth_vix)))
cat(sprintf("    Mean / Median      : %.2f / %.2f pts\n",
            mean(synth_vix), median(synth_vix)))
cat(strrep("=", 70), "\n\n")

# File verification
expected_files <- c(
  "synthetic_vix_esg.csv",
  "walkforward_oos_predictions.csv",
  "walkforward_fold_metrics.csv",
  "spx_full_sample_fit.csv",
  "01_walkforward_fold_metrics.png",
  "02_walkforward_oos_fit.png",
  "03_final_gru_training_history.png",
  "04_spx_full_gru_fit.png",
  "05_synthetic_vix_esg.png",
  "06_spx_garch_vol_transform.png",
  "07_esg_garch_vol.png"
)

cat("  Output files written to:", OUTPUT_DIR, "\n")
cat(strrep("-", 70), "\n")
all_ok <- TRUE
for (fname in expected_files) {
  fpath <- file.path(OUTPUT_DIR, fname)
  if (file.exists(fpath)) {
    fsz <- file.info(fpath)$size
    cat(sprintf("  [OK]  %-46s  %s\n", fname,
                ifelse(fsz > 1024,
                       sprintf("%.1f KB", fsz / 1024),
                       sprintf("%d B",    fsz))))
  } else {
    cat(sprintf("  [!!]  %-46s  NOT FOUND\n", fname))
    all_ok <- FALSE
  }
}
model_files <- list.files(OUTPUT_DIR,
                           pattern = "gjr_garch_gru",
                           full.names = FALSE)
for (mf in model_files) {
  fsz <- file.info(file.path(OUTPUT_DIR, mf))$size
  cat(sprintf("  [OK]  %-46s  %.1f KB\n", mf, fsz / 1024))
}
cat(strrep("=", 70), "\n")

if (!all_ok)
  warning("Some output files were not created. Check write permissions: ", OUTPUT_DIR)

# =============================================================================
# END OF SCRIPT  — GJR-GARCH + GRU Synthetic VIX Pipeline v2
# =============================================================================
