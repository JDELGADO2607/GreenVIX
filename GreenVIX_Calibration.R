library(readr)
library(Rsolnp)     # solnp() — Sequential Quadratic Programming
library(ggplot2)    # publication-quality diagnostics
library(gridExtra)  # multi-panel plot layout
library(scales) 
library(readxl)
df <- read_excel("C:/Users/juanm/Downloads/Base Definitiva VIX Verde.xlsx", sheet = "Synthetic")

spx <- c(as.numeric(na.omit(df$SPX)))
r_spx <- c(na.omit(diff(log(spx))))
vix <- c(as.numeric(na.omit(df$VIX)))

vix_horizon = 22

vix_weight = 1.0

tdpy = 252 #Trading Days Per Year

verbose = 1

max_iter = 500
max_inner_iter = 100

date <- c(as.Date(na.omit(df$Date)))

data <- data.frame(date[-1], spx[-1], r_spx, vix[-1])

vix_affine_coef <- function(omega, alpha, beta, gamma_star, T_H, ann_factor) {
  
  phi <- beta + alpha * gamma_star^2
  alpha_q <- alpha * gamma_star^2
  
  A_tau <- 0
  B_tau <- 1
  
  sumA <- 0
  sumB <- 0
  
  for (tau in seq_len(T_H)) {
    
    A_new <- A_tau + omega * B_tau + alpha
    B_new <- phi * B_tau + alpha_q
    
    sumA <- sumA + A_new
    sumB <- sumB + B_new
    
    A_tau <- A_new
    B_tau <- B_new
    
  }
  
  list(sumA = sumA, sumB = sumB, phi_Q = phi)
}

joint_hn_nll <- function(params,
                         returns,       # SPX log-returns (decimal)
                         vix_obs,       # observed VIX index levels
                         T_H,           # VIX horizon (trading days)
                         ann_factor,    # 252
                         vix_weight) {  # w
  
  omega   <- params[1]
  alpha   <- params[2]
  beta    <- params[3]
  gamma   <- params[4]
  lambda  <- params[5]
  sigma_v <- params[6]  
  
  T_obs <- length(returns)
  
  # ── Hard constraint guard (duplicates solnp inequality check as a safety net)
  if (omega <= 0 || alpha <= 0 || beta <= 0 || sigma_v <= 0) return(1e10)
  
  # Stationarity check
  stat_lhs <- beta + alpha * gamma^2
  if (stat_lhs >= 1) return(1e10)
  
  # ── Risk-neutral leverage ────────────────────────────────────────────────────
  gamma_star <- gamma + lambda + 0.5
  
  vix_coef <- vix_affine_coef(
    omega, alpha, beta, gamma_star, T_H, ann_factor
  )
  
  # If Q-measure persistence ≥ 1 the forward variance diverges → penalise
  if (vix_coef$phi_Q >= 1) return(1e10)
  
  sumA <- vix_coef$sumA
  sumB <- vix_coef$sumB
  
  # Scalar multiplier converting h_t → model VIX² (annualised %-squared)
  # VIX²_hat = (ann_factor / T_H) · (sumA + sumB · h_t) · 10000
  vix_ann_mult <- (ann_factor / T_H) * 10000
  
  # ── Initialise variance at unconditional (stationary) level ─────────────────
  #    E[h] = (ω + α) / (1 − β − α·γ²)
  denom_stat <- 1 - stat_lhs
  h <- (omega + alpha) / denom_stat
  if (!is.finite(h) || h <= 0) return(1e10)
  
  # ── Variance of VIX log-likelihood (σ²_v) ───────────────────────────────────
  sigma2_v <- sigma_v^2
  
  # ── Accumulate joint log-likelihood ─────────────────────────────────────────
  ll_spx <- 0
  ll_vix <- 0
  
  for (t in seq_along(returns)) {
    
    # ── SPX leg: conditional density r_t | F_{t-1} ~ N(λ·h_t, h_t) ──────────
    mu_t    <- lambda * h
    resid_t <- returns[t] - mu_t
    
    # −½ [log(h_t) + ε²/h_t]   (drop constant −½·log(2π))
    ll_spx <- ll_spx - 0.5 * (log(h) + resid_t^2 / h)
    
    # ── VIX leg: log-VIX Gaussian measurement equation ────────────────────────
    # Model-implied VIX² in annualised percentage-squared units
    vix2_hat <- vix_ann_mult * (sumA + sumB * h)
    
    if (vix2_hat <= 0) return(1e10)   # affine recursion blew up
    
    # Observed log(VIX) and model-implied ½·log(VIX²_hat) = log(VIX_hat)
    log_vix_obs  <- log(vix_obs[t])        # log of raw VIX level
    log_vix_hat  <- 0.5 * log(vix2_hat)   # ½·log(VIX²) = log(VIX)
    
    vix_innov <- log_vix_obs - log_vix_hat
    
    # −½ [log(σ²_v) + innovation²/σ²_v]
    ll_vix <- ll_vix - 0.5 * (log(sigma2_v) + vix_innov^2 / sigma2_v)
    
    # ── P-measure variance recursion ──────────────────────────────────────────
    # z_t = (r_t − λ·h_t) / √h_t   (standardised innovation)
    z_t <- resid_t / sqrt(h)
    h   <- omega + beta * h + alpha * (z_t - gamma * sqrt(h))^2
    
    if (!is.finite(h) || h <= 0) return(1e10)
  }
  
  # ── Joint panel NLL ─────────────────────────────────────────────────────────
  nll <- -(ll_spx + vix_weight * ll_vix)
  
  # Final finiteness guard
  if (!is.finite(nll)) return(1e10)
  
  return(nll)
  
}

params_init <- c(
  omega   = 1e-6,   # ω₀
  alpha   = 1e-6,   # α₀
  beta    = 0.90,   # β₀
  gamma   = 100,    # γ₀
  lambda  = 2.0,    # λ₀
  sigma_v = 0.10    # σ_v₀
)

nll_init <- joint_hn_nll(
  params      = params_init,
  returns     = r_spx,
  vix_obs     = vix,
  T_H         = vix_horizon,
  ann_factor  = tdpy,
  vix_weight  = vix_weight
)

# Unconditional variance at starting params (used as sanity check)
stat_lhs_init <- params_init["beta"] + params_init["alpha"] * params_init["gamma"]^2
h_uncond_init <- (params_init["omega"] + params_init["alpha"]) /
  (1 - stat_lhs_init)

# ── Lower and upper bounds ─────────────────────────────────────────────────────
#             omega   alpha   beta    gamma   lambda  sigma_v
lb <- c(      1e-9,   1e-9,   1e-6,   0,      -50,    1e-6  )
ub <- c(      1e-1,   1e-1,   0.9999, 1000,    50,    5.0   )

# ── Inequality constraint: stationarity ───────────────────────────────────────
# g(θ) = β + α·γ²  must satisfy  0 < g(θ) < 1
ineq_fun <- function(params, ...) {
  alpha <- params[2]
  beta  <- params[3]
  gamma <- params[4]
  beta + alpha * gamma^2
}

ineq_lb <- 1e-9      # enforce strict positivity of stationary component
ineq_ub <- 1 - 1e-6  # strict upper bound away from unit root

set.seed(2024)   # reproducibility (solnp is deterministic but good practice)

t_start <- proc.time()

opt_result <- solnp(
  pars     = params_init,
  
  # Objective: negative joint log-likelihood (scalar)
  fun      = joint_hn_nll,
  
  # Inequality constraints: ineqLB ≤ ineqfun(pars) ≤ ineqUB
  ineqfun  = ineq_fun,
  ineqLB   = ineq_lb,
  ineqUB   = ineq_ub,
  
  # Box constraints (element-wise)
  LB       = lb,
  UB       = ub,
  
  # Pass fixed data to the objective
  returns    = r_spx,
  vix_obs    = vix,
  T_H        = vix_horizon,
  ann_factor = tdpy,
  vix_weight = vix_weight,
  
  # Solver control
  control  = list(
    trace   = verbose,
    maxit   = max_iter,
    rho     = 1,           # augmented-Lagrangian penalty weight
    delta   = 1e-7,        # gradient finite-difference step size
    tol     = 1e-8,        # convergence tolerance
    inner.iter = max_inner_iter
  )
)

t_elapsed <- proc.time() - t_start

pars_opt <- opt_result$pars
names(pars_opt) <- c("omega", "alpha", "beta", "gamma", "lambda", "sigma_v")

omega_hat   <- pars_opt["omega"]
alpha_hat   <- pars_opt["alpha"]
beta_hat    <- pars_opt["beta"]
gamma_hat   <- pars_opt["gamma"]
lambda_hat  <- pars_opt["lambda"]
sigma_v_hat <- pars_opt["sigma_v"]

# ── Derived physical quantities ───────────────────────────────────────────────
stat_measure   <- beta_hat + alpha_hat * gamma_hat^2
h_uncond_hat   <- (omega_hat + alpha_hat) / (1 - stat_measure)
vol_uncond_ann <- sqrt(h_uncond_hat * tdpy) * 100

# ── Risk-neutral leverage γ* = γ + λ + 0.5 ────────────────────────────────────
gamma_star_hat <- gamma_hat + lambda_hat + 0.5

# Q-measure persistence
phi_Q <- beta_hat + alpha_hat * gamma_star_hat^2

vix_coef_opt <- vix_affine_coef(
  omega      = omega_hat,
  alpha      = alpha_hat,
  beta       = beta_hat,
  gamma_star = gamma_star_hat,
  T_H        = vix_horizon,
  ann_factor = tdpy
)

sumA_opt <- vix_coef_opt$sumA
sumB_opt <- vix_coef_opt$sumB
vix_ann_mult_opt <- (tdpy / vix_horizon) * 10000

# Initialise filter
h_path    <- numeric(length(r_spx))
vix_model <- numeric(length(r_spx))

h_curr <- (omega_hat + alpha_hat) / (1 - stat_measure)   # h_0 = unconditional

for (t in seq_along(r_spx)) {
  h_path[t] <- h_curr
  
  # Model-implied VIX level  = √(VIX²_hat)
  vix2_hat_t    <- vix_ann_mult_opt * (sumA_opt + sumB_opt * h_curr)
  vix_model[t]  <- sqrt(max(vix2_hat_t, 0))
  
  # Variance recursion
  mu_t   <- lambda_hat * h_curr
  resid  <- r_spx[t] - mu_t
  z_t    <- resid / sqrt(h_curr)
  h_curr <- omega_hat + beta_hat * h_curr +
    alpha_hat * (z_t - gamma_hat * sqrt(h_curr))^2
}

# Annualised physical volatility path (%)
vol_path_ann <- sqrt(h_path * tdpy) * 100

# VIX fit error statistics
vix_error     <- vix_model - vix[-1]
vix_rmse      <- sqrt(mean(vix_error^2))
vix_mae       <- mean(abs(vix_error))
vix_corr      <- cor(vix_model, vix[-1])

cat(sprintf("  VIX fit RMSE      : %.4f VIX points\n", vix_rmse))
cat(sprintf("  VIX fit MAE       : %.4f VIX points\n", vix_mae))
cat(sprintf("  VIX correlation   : %.6f\n", vix_corr))
cat(sprintf("  Mean VIX bias     : %.4f (model − observed)\n", mean(vix_error)))


# ==============================================================================
# 10.  DIAGNOSTIC PLOTS
# ==============================================================================

cat("\n", strrep("=", 70), "\n")
cat("  STEP 6 — Generating diagnostic plots\n")
cat(strrep("=", 70), "\n\n")

plot_df <- data.frame(
  Date        = date[-1],
  LogReturn   = r_spx,
  VIX_obs     = vix[-1],
  VIX_model   = vix_model,
  Vol_ann     = vol_path_ann,
  VIX_error   = vix_error
)

# Colour palette (accessible)
col_obs   <- "#2166AC"   # blue  — observed
col_model <- "#D6604D"   # red   — model
col_vol   <- "#4DAC26"   # green — physical vol

# ── Plot 1: Observed vs Model VIX ─────────────────────────────────────────────
p1 <- ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = VIX_obs,   colour = "Observed VIX"), linewidth = 0.5, alpha = 0.9) +
  geom_line(aes(y = VIX_model, colour = "Model VIX"),    linewidth = 0.5, alpha = 0.9) +
  scale_colour_manual(values = c("Observed VIX" = col_obs, "Model VIX" = col_model)) +
  labs(
    title    = "Observed vs. Model-Implied VIX",
    subtitle = sprintf("T_H = %d days | RMSE = %.3f | Corr = %.4f",
                       vix_horizon, vix_rmse, vix_corr),
    x        = NULL,
    y        = "VIX Level",
    colour   = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

# ── Plot 2: Physical annualised volatility √(h_t·252) × 100 ──────────────────
p2 <- ggplot(plot_df, aes(x = Date, y = Vol_ann)) +
  geom_line(colour = col_vol, linewidth = 0.5) +
  labs(
    title    = "Physical Conditional Volatility  √(h_t · 252) × 100",
    subtitle = "Heston-Nandi GARCH filter (P-measure)",
    x        = NULL,
    y        = "Annualised Volatility (%)"
  ) +
  theme_minimal(base_size = 11)

# ── Plot 3: VIX pricing error time series ────────────────────────────────────
p3 <- ggplot(plot_df, aes(x = Date, y = VIX_error)) +
  geom_line(colour = "grey40", linewidth = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "black") +
  geom_hline(yintercept = c(-vix_rmse, vix_rmse),
             linetype = "dotted", colour = col_model) +
  labs(
    title    = "VIX Pricing Error  (Model − Observed)",
    subtitle = sprintf("MAE = %.3f | ±RMSE bands shown in red", vix_mae),
    x        = NULL,
    y        = "Error (VIX points)"
  ) +
  theme_minimal(base_size = 11)

# ── Plot 4: Scatter — model vs observed VIX ──────────────────────────────────
p4 <- ggplot(plot_df, aes(x = VIX_obs, y = VIX_model)) +
  geom_point(alpha = 0.3, size = 0.8, colour = col_obs) +
  geom_abline(slope = 1, intercept = 0, colour = "black", linetype = "dashed") +
  labs(
    title    = "Model vs. Observed VIX (Scatter)",
    subtitle = sprintf("R² = %.4f", vix_corr^2),
    x        = "Observed VIX",
    y        = "Model-Implied VIX"
  ) +
  theme_minimal(base_size = 11)

# ── Combine and save ──────────────────────────────────────────────────────────
combined_plot <- gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)

output_plot_path <- "hn_garch_diagnostics.png"
ggsave(
  filename = output_plot_path,
  plot     = combined_plot,
  width    = 14,
  height   = 10,
  dpi      = 180
)

# ==============================================================================
# 11.  SUMMARY REPORT
# ==============================================================================

cat("\n", strrep("=", 70), "\n")
cat("  FINAL SUMMARY — Heston-Nandi Joint MLE Calibration\n")
cat(strrep("=", 70), "\n\n")

summary_list <- list(
  
  # ── Sample info ──────────────────────────────────────────────────────────────
  sample_start     = as.character(min(date[-1])),
  sample_end       = as.character(max(date[-1])),
  n_observations   = length(r_spx),
  vix_horizon_days = vix_horizon,
  
  # ── Optimisation ─────────────────────────────────────────────────────────────
  convergence_code = opt_result$convergence,
  final_nll        = opt_result$values[length(opt_result$values)],
  elapsed_seconds  = unname(t_elapsed["elapsed"]),
  
  # ── Physical parameters ───────────────────────────────────────────────────────
  omega            = unname(omega_hat),
  alpha            = unname(alpha_hat),
  beta             = unname(beta_hat),
  gamma            = unname(gamma_hat),
  lambda           = unname(lambda_hat),
  sigma_v          = unname(sigma_v_hat),
  
  # ── Derived physical quantities ───────────────────────────────────────────────
  stationarity_lhs = unname(stat_measure),
  h_unconditional  = unname(h_uncond_hat),
  vol_unconditional_ann_pct = unname(vol_uncond_ann),
  
  # ── Risk-neutral parameters ───────────────────────────────────────────────────
  gamma_star       = unname(gamma_star_hat),
  phi_Q            = unname(phi_Q),
  
  # ── VIX fit quality ───────────────────────────────────────────────────────────
  vix_rmse         = vix_rmse,
  vix_mae          = vix_mae,
  vix_correlation  = vix_corr,
  vix_R_squared    = vix_corr^2
)

# ── Save as RDS (for downstream option pricing / risk-neutral MC) ─────────────
saveRDS(summary_list, file = "hn_garch_calibration_results.rds")
cat("  Results saved to: hn_garch_calibration_results.rds\n\n")

# ── Pretty print ──────────────────────────────────────────────────────────────
cat("  ╔══════════════════════════════════════════════════════════╗\n")
cat("  ║  PHYSICAL PARAMETERS                                     ║\n")
cat(sprintf("  ║    ω  = %12.6e                                    ║\n", omega_hat))
cat(sprintf("  ║    α  = %12.6e                                    ║\n", alpha_hat))
cat(sprintf("  ║    β  = %12.8f                                    ║\n", beta_hat))
cat(sprintf("  ║    γ  = %12.6f                                    ║\n", gamma_hat))
cat(sprintf("  ║    λ  = %12.6f                                    ║\n", lambda_hat))
cat(sprintf("  ║    σ_v= %12.6f                                    ║\n", sigma_v_hat))
cat("  ╠══════════════════════════════════════════════════════════╣\n")
cat("  ║  STATIONARITY CHECK                                      ║\n")
cat(sprintf("  ║    β + α·γ² = %.8f  < 1  ✓                        ║\n", stat_measure))
cat(sprintf("  ║    Uncond. vol (ann.) = %.4f%%                     ║\n", vol_uncond_ann))
cat("  ╠══════════════════════════════════════════════════════════╣\n")
cat("  ║  RISK-NEUTRAL LEVERAGE  (key for option pricing)         ║\n")
cat(sprintf("  ║    γ* = γ + λ + 0.5 = %.6f                         ║\n", gamma_star_hat))
cat(sprintf("  ║    Q-persistence φ_Q = %.8f                        ║\n", phi_Q))
cat("  ╠══════════════════════════════════════════════════════════╣\n")
cat("  ║  VIX FIT QUALITY                                         ║\n")
cat(sprintf("  ║    RMSE        = %.4f VIX points                   ║\n", vix_rmse))
cat(sprintf("  ║    MAE         = %.4f VIX points                   ║\n", vix_mae))
cat(sprintf("  ║    Correlation = %.6f                              ║\n", vix_corr))
cat(sprintf("  ║    R²          = %.6f                              ║\n", vix_corr^2))
cat("  ╚══════════════════════════════════════════════════════════╝\n\n")

cat("  ✔  Calibration complete.\n\n")

