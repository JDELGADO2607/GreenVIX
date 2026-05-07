# GreenVIX Calibration — Technical Documentation

**Heston-Nandi (2000) Affine GARCH · Joint SPX/VIX MLE · ESG Synthetic VIX**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Mathematical Framework](#3-mathematical-framework)
   - 3.1 [Physical Measure (P) Dynamics](#31-physical-measure-p-dynamics)
   - 3.2 [Risk-Neutral Measure (Q) Transformation](#32-risk-neutral-measure-q-transformation)
   - 3.3 [Affine Variance Recursion for VIX](#33-affine-variance-recursion-for-vix)
   - 3.4 [Characteristic Function](#34-characteristic-function)
   - 3.5 [Gil-Pelaez Option Pricing Inversion](#35-gil-pelaez-option-pricing-inversion)
   - 3.6 [CBOE Variance Swap Formula](#36-cboe-variance-swap-formula)
   - 3.7 [Dual-Maturity VIX Interpolation](#37-dual-maturity-vix-interpolation)
4. [C++ Pricer — `hn_pricer.cpp`](#4-c-pricer--hn_pricercpp)
   - 4.1 [Dependencies and Compilation](#41-dependencies-and-compilation)
   - 4.2 [Function: `hn_call_price`](#42-function-hn_call_price)
   - 4.3 [Function: `hn_put_price`](#43-function-hn_put_price)
   - 4.4 [Function: `hn_option_strip`](#44-function-hn_option_strip)
   - 4.5 [Numerical Integration Design](#45-numerical-integration-design)
   - 4.6 [Characteristic Function Recursion](#46-characteristic-function-recursion)
5. [R Script — `main.R`](#5-r-script--mainr)
   - 5.1 [Section 0 — Packages](#51-section-0--packages)
   - 5.2 [Section 1 — Settings](#52-section-1--settings)
   - 5.3 [Section 2 — Data Loading](#53-section-2--data-loading)
   - 5.4 [Section 3 — Affine Coefficients](#54-section-3--affine-coefficients)
   - 5.5 [Section 4 — Joint NLL Function](#55-section-4--joint-nll-function)
   - 5.6 [Section 5 — SPX/VIX MLE](#56-section-5--spxvix-mle)
   - 5.7 [Section 6 — SPX Diagnostic Plots](#57-section-6--spx-diagnostic-plots)
   - 5.8 [Section 7 — ESG Physical MLE](#58-section-7--esg-physical-mle)
   - 5.9 [Section 8 — ESG P-Measure Filter](#59-section-8--esg-p-measure-filter)
   - 5.10 [Section 9 — CBOE Variance Formula](#510-section-9--cboe-variance-formula)
   - 5.11 [Section 10 — Strike Grid Builder](#511-section-10--strike-grid-builder)
   - 5.12 [Section 11 — Parallelised Daily VIX Loop](#512-section-11--parallelised-daily-vix-loop)
   - 5.13 [Section 12 — ESG Output](#513-section-12--esg-output)
6. [Parameter Reference](#6-parameter-reference)
7. [Calibration Constraints](#7-calibration-constraints)
8. [Data Requirements](#8-data-requirements)
9. [Outputs](#9-outputs)
10. [Design Decisions and Known Limitations](#10-design-decisions-and-known-limitations)
11. [References](#11-references)

---

## 1. Project Overview

This codebase implements a full quantitative pipeline for constructing a **synthetic volatility index (GreenVIX)** for a custom ESG equity index that has no traded options. The pipeline has three stages:

| Stage | Description | Key Output |
|---|---|---|
| **SPX/VIX Joint MLE** | Calibrate HN-GARCH parameters against both SPX log-returns and observed VIX levels simultaneously | `{ω, α, β, γ, λ, σ_v}` SPX |
| **ESG Physical MLE** | Calibrate ESG variance dynamics from ESG log-returns alone | `{ω, α, β, γ}` ESG |
| **Synthetic VIX** | Price a dense theoretical option grid under the Q-measure using the HN closed form; apply the CBOE variance swap formula | `vix_esg` time series |

The core insight enabling Stage 3 is that the **variance risk premium** `λ` (which bridges P and Q) is not identifiable from ESG returns alone but is shared with the broad market. We borrow `λ_SPX` from Stage 1 and apply it to the ESG index via the risk-neutral leverage transformation `γ*_ESG = γ_ESG + λ_SPX + 0.5`.

---

## 2. Repository Structure

```
.
├── hn_pricer.cpp                  # Rcpp C++ option pricer (Gil-Pelaez inversion)
├── main.R                         # Full calibration pipeline
├── hn_garch_spx_results.rds       # SPX/VIX MLE output (generated)
├── hn_garch_esg_results.rds       # ESG calibration output (generated)
├── esg_synthetic_vix.csv          # Daily GreenVIX time series (generated)
├── hn_garch_spx_diagnostics.png   # SPX fit diagnostic plots (generated)
└── hn_garch_esg_diagnostics.png   # ESG volatility plots (generated)
```

**Input required:** a single Excel workbook with sheet `"Synthetic"` containing columns:

| Column | Description |
|---|---|
| `Date` | Trading dates |
| `SPX` | S&P 500 price index levels |
| `VIX` | CBOE VIX index levels (raw, e.g. `18.5`) |
| `Price (Adjusted BESG)` | ESG index adjusted price levels |

---

## 3. Mathematical Framework

### 3.1 Physical Measure (P) Dynamics

The **Heston-Nandi (2000) GARCH** model specifies the following discrete-time system under the physical measure P:

**Return equation:**

$$r_t \equiv \ln\left(\frac{S_t}{S_{t-1}}\right) = \lambda h_t + \sqrt{h_t}\, z_t, \qquad z_t \sim \mathcal{N}(0,1) \text{ i.i.d.}$$

where the daily risk-free rate is set to zero (`r = 0`) for the calibration step.

**Conditional variance recursion:**

$$h_t = \omega + \beta h_{t-1} + \alpha\left(z_{t-1} - \gamma\sqrt{h_{t-1}}\right)^2$$

**Unconditional (stationary) variance:**

$$\bar{h} = \frac{\omega + \alpha}{1 - \beta - \alpha\gamma^2}$$

This is used to initialise the filter at `h_0 = h̄`.

**Stationarity condition:**

$$\beta + \alpha\gamma^2 < 1$$

This is a hard constraint enforced during both SPX and ESG optimisation.

---

### 3.2 Risk-Neutral Measure (Q) Transformation

The risk-neutral leverage parameter is obtained by an **exact measure change** (Heston & Nandi 2000, Proposition 1):

$$\gamma^* = \gamma + \lambda + \frac{1}{2}$$

Under Q, the variance recursion takes the same GARCH form but with `γ` replaced by `γ*`. This is the affine property that makes the model tractable.

**ESG-specific application:** because the ESG index has no traded options, `λ` cannot be identified from ESG data. We transplant the broad market premium:

$$\gamma^*_{\text{ESG}} = \gamma_{\text{ESG}} + \lambda_{\text{SPX}} + 0.5$$

The justification is that the variance risk premium is a **market-wide price of volatility risk**, not an asset-specific quantity. This is the identification assumption of the methodology.

---

### 3.3 Affine Variance Recursion for VIX

Because the HN-GARCH model is affine in `h_t`, the Q-measure expected future variance is linear in the current state:

$$E^Q_t\left[h_{t+\tau}\right] = A(\tau) + B(\tau) \cdot h_t$$

with **forward recursion** initialised at `A(0) = 0, B(0) = 1`:

$$A(\tau) = A(\tau-1) + \omega \cdot B(\tau-1) + \alpha$$

$$B(\tau) = \underbrace{(\beta + \alpha{\gamma^*}^2)}_{\phi_Q} \cdot B(\tau-1) + \alpha{\gamma^*}^2$$

The **Q-measure persistence** is `φ_Q = β + αγ*²`. The affine recursion is only meaningful when `φ_Q < 1` (a second hard constraint).

**Model-implied VIX²** (annualised, percentage-squared, CBOE units):

$$\widehat{\text{VIX}}_t^2 = \frac{252}{T_H} \cdot \left(\sum_{\tau=1}^{T_H} A(\tau) + \sum_{\tau=1}^{T_H} B(\tau) \cdot h_t\right) \times 10000$$

where `T_H = 22` trading days. The `× 10000` factor converts from variance in decimal² to percentage².

---

### 3.4 Characteristic Function

Under Q, the log-price `log(S_T/S_0)` has the moment-generating function:

$$f(\phi) = E^Q_0\left[e^{i\phi \ln S_T}\right] = \exp\!\left(A(\phi, T) + B(\phi, T) \cdot h_0 + i\phi \ln S_0\right)$$

where the **complex-valued coefficients** satisfy the backward recursion (run forward from `τ = 0` to `τ = T_days`):

$$A_{\text{next}} = A + \phi r + B\omega - \frac{1}{2}\ln(1 - 2\alpha B)$$

$$B_{\text{next}} = \phi(\lambda + \gamma) - \frac{1}{2}\gamma^2 + \beta B + \frac{\frac{1}{2}(\phi - \gamma)^2}{1 - 2\alpha B}$$

> **Note on signs:** the recursion above uses the **physical-measure** parameterisation of the CF as presented in Heston & Nandi (2000), eq. (13). The `γ` and `λ` here are the Q-measure parameters; i.e., when called for option pricing, `gamma_star` replaces `gamma` and `lambda` plays the role of the expected return under Q (set to `r_day` in practice).

The recursion is singular when `Re(1 - 2αB) ≤ 0`. The C++ implementation guards this with an early-exit check.

---

### 3.5 Gil-Pelaez Option Pricing Inversion

Call price by the **Gil-Pelaez (1951)** inversion theorem:

$$C = S \cdot P_1 - K e^{-rT} \cdot P_2$$

where the two risk-adjusted probabilities are recovered from the characteristic function:

$$P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re}\!\left[\frac{e^{-i\phi \ln K} \cdot f_j(\phi)}{i\phi}\right] d\phi$$

- `f_2(φ) = f(φ)` — risk-neutral CF
- `f_1(φ) = f(φ - i) / f(-i)` — share-measure CF (tilted by `φ → φ - i`)

**Put price** via put-call parity (no additional inversion needed):

$$P = C - S + K e^{-rT}$$

---

### 3.6 CBOE Variance Swap Formula

For a set of OTM strikes `{K_i}` with option prices `{Q_i}`, the CBOE variance is:

$$\sigma^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{RT} Q_i - \frac{1}{T}\left(\frac{F}{K_0} - 1\right)^2$$

where:

- `F = S · e^{RT}` — forward price
- `K_0` — highest strike satisfying `K_0 ≤ F`
- `ΔK_i` — central difference for interior strikes, one-sided at the wings:

$$\Delta K_i = \begin{cases} K_2 - K_1 & i = 1 \\ K_n - K_{n-1} & i = n \\ \dfrac{K_{i+1} - K_{i-1}}{2} & \text{otherwise} \end{cases}$$

**OTM selection rule:**

$$Q_i = \begin{cases} \text{Put}(K_i) & K_i < K_0 \\ \frac{1}{2}\left[\text{Put}(K_0) + \text{Call}(K_0)\right] & K_i = K_0 \\ \text{Call}(K_i) & K_i > K_0 \end{cases}$$

**Tail truncation (CBOE rule):** starting from each wing and scanning inward, discard all strikes once **two consecutive** option prices fall below `PRICE_FLOOR = 0.001`. This replicates the CBOE's two-consecutive-zero-bid termination rule in a theoretical (no-zero-bid) environment.

---

### 3.7 Dual-Maturity VIX Interpolation

Two option grids are constructed for each day: near-term (`T1 = 23/365`) and next-term (`T2 = 37/365`). The variances `σ²₁` and `σ²₂` are interpolated to the target horizon:

$$\sigma^2_{30} = \left[T_1 \sigma^2_1 \cdot \frac{N_2 - N_{\text{target}}}{N_2 - N_1} + T_2 \sigma^2_2 \cdot \frac{N_{\text{target}} - N_1}{N_2 - N_1}\right] \cdot \frac{365}{N_{\text{target}}}$$

where `N_1 = 23`, `N_2 = 37`, `N_target = 30` are calendar day counts, and `T_1`, `T_2` are the corresponding year fractions. The final GreenVIX level is:

$$\text{GreenVIX}_t = 100 \cdot \sqrt{\sigma^2_{30}}$$

---

## 4. C++ Pricer — `hn_pricer.cpp`

### 4.1 Dependencies and Compilation

```cpp
#include <Rcpp.h>
#include <complex>
#include <cmath>
using namespace Rcpp;
using namespace std::complex_literals;
```

The file is compiled from R via:

```r
Rcpp::sourceCpp("hn_pricer.cpp")
```

This triggers `clang++` (or `g++`) with the flags set by R's `Makeconf`. No manual compilation step is required. All three exported functions become available in the R session immediately after `sourceCpp`.

**Compiler requirement:** C++11 or later (required for `<complex>` and lambda syntax). `Rcpp` sets this automatically.

---

### 4.2 Function: `hn_call_price`

```cpp
// [[Rcpp::export]]
double hn_call_price(double S, double K, double T, double r,
                     double h0, double omega, double alpha,
                     double beta, double gamma, double lambda)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `S` | `double` | Current underlying price |
| `K` | `double` | Strike price |
| `T` | `double` | Time to expiry in **year fractions** (e.g. `23/365`) |
| `r` | `double` | Continuously compounded annual risk-free rate |
| `h0` | `double` | Current conditional variance (from P-measure filter) |
| `omega` | `double` | Variance intercept ω |
| `alpha` | `double` | ARCH coefficient α |
| `beta` | `double` | GARCH coefficient β |
| `gamma` | `double` | Leverage parameter — **pass `gamma_star` for Q-measure pricing** |
| `lambda` | `double` | Variance risk premium λ |

**Returns:** European call price as `double`. Floor at zero enforced.

**Internal steps:**

1. Convert `T` to trading days: `T_days = round(T × 252)`
2. Convert annual rate to daily: `r_day = r / 252`
3. Evaluate characteristic function at `φ` (for P2) and `φ - i` (for P1) via `log_cf` lambda
4. Integrate with trapezoidal rule on `[dphi, upper_lim]` in 2000 steps
5. Recover `P1`, `P2`; clamp to `[0, 1]`
6. Return `max(S·P1 - K·exp(-rT)·P2, 0)`

---

### 4.3 Function: `hn_put_price`

```cpp
// [[Rcpp::export]]
double hn_put_price(double S, double K, double T, double r,
                    double h0, double omega, double alpha,
                    double beta, double gamma, double lambda)
```

Calls `hn_call_price` internally and applies **put-call parity**:

$$P = C - S + K e^{-rT}$$

No additional numerical integration. This guarantees exact put-call parity by construction and avoids a second numerical inversion.

---

### 4.4 Function: `hn_option_strip`

```cpp
// [[Rcpp::export]]
NumericVector hn_option_strip(NumericVector K_vec, double S, double T,
                              double r, double h0, double omega,
                              double alpha, double beta, double gamma,
                              double lambda, std::string type = "call")
```

**Vectorised pricer** over a strike vector `K_vec`. Takes identical scalar parameters to `hn_call_price` plus a `type` argument (`"call"` or `"put"`).

This is the primary entry point used in `main.R`. Replacing the per-strike R `for` loop with a single call to `hn_option_strip` eliminates R interpreter overhead for the entire 150-strike grid, which is the innermost loop of the computation.

---

### 4.5 Numerical Integration Design

The Gil-Pelaez integral is evaluated using the **midpoint trapezoidal rule** on a uniform grid:

```cpp
int    N     = 2000;
double upper = 100.0;
double dphi  = upper / N;   // step = 0.05
```

Quadrature points: `φ_j = (j - 0.5) · dphi` for `j = 1, …, 2000`.

**Why these choices:**

- `upper = 100` is sufficient because the HN-GARCH CF decays rapidly for `φ >> 1/√h_0`. For typical equity variance (`h_0 ~ 1e-4`), the integrand is numerically zero by `φ = 50`.
- `N = 2000` gives step `dphi = 0.05`, which is fine enough to avoid discretisation error at the peak of the integrand near `φ ≈ 1`.
- The midpoint rule on a uniform grid is equivalent to the trapezoidal rule and has `O(dphi²)` error per unit interval.

---

### 4.6 Characteristic Function Recursion

The `log_cf` lambda inside `hn_call_price` runs the following loop using `std::complex<double>`:

```cpp
for (int tau = 0; tau < T_days; ++tau) {
    denom  = 1.0 - 2.0 * alpha * B;
    A_next = A + phi*r_day + B*omega - 0.5*log(denom);
    B_next = phi*(lambda + gamma) - 0.5*gamma*gamma
             + beta*B + 0.5*(phi - gamma)^2 / denom;
    A = A_next; B = B_next;
}
return exp(A + B*h0 + i*phi*log(S));
```

The guard `if (Re(denom) ≤ 1e-10)` triggers an early return of `NA_complex_`, which the calling integrand converts to `0.0` (contributing nothing to the integral). This prevents `log(0)` and division-by-zero without aborting the entire pricing call.

---

## 5. R Script — `main.R`

### 5.1 Section 0 — Packages

```r
required_pkgs <- c("readxl", "Rsolnp", "ggplot2", "gridExtra",
                   "scales", "parallel", "Rcpp")
```

| Package | Role |
|---|---|
| `readxl` | Read `.xlsx` data file |
| `Rsolnp` | Sequential Quadratic Programming optimiser |
| `ggplot2` | Diagnostic plots |
| `gridExtra` | Multi-panel plot layout |
| `scales` | Axis formatting |
| `parallel` | Multi-core daily VIX loop |
| `Rcpp` | Load and compile `hn_pricer.cpp` |

`Rcpp::sourceCpp("hn_pricer.cpp")` is called immediately after package loading. This compiles the C++ file and registers `hn_call_price`, `hn_put_price`, and `hn_option_strip` in the R session.

---

### 5.2 Section 1 — Settings

All user-facing configuration is consolidated at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | *(user path)* | Path to Excel workbook |
| `SHEET_NAME` | `"Synthetic"` | Worksheet name |
| `VIX_HORIZON` | `22` | Trading days for affine VIX formula |
| `VIX_WEIGHT` | `1.0` | Weight `w` on VIX leg in joint NLL |
| `TDPY` | `252` | Trading days per year |
| `VERBOSE` | `1` | `solnp` trace level |
| `MAX_ITER` | `500` | SQP outer iteration limit |
| `MAX_INNER` | `100` | BFGS inner iteration limit |
| `N_STRIKES` | `150` | Strike grid density |
| `M_MIN` | `0.01` | Minimum moneyness `K/S` |
| `M_MAX` | `2.01` | Maximum moneyness `K/S` |
| `PRICE_FLOOR` | `0.001` | Tail truncation threshold |
| `T1_DAYS` | `23` | Near-term expiry (calendar days) |
| `T2_DAYS` | `37` | Next-term expiry (calendar days) |
| `TARGET_DAYS` | `30` | VIX interpolation target (calendar days) |
| `R_F_ANNUAL` | `0` | Annual risk-free rate |
| `N_CORES` | `detectCores() - 1` | Worker count for parallel loop |

---

### 5.3 Section 2 — Data Loading

```r
df      <- read_excel(DATA_PATH, sheet = SHEET_NAME)
r_spx   <- diff(log(spx))          # SPX log returns, length N-1
vix[-1]                             # VIX aligned to returns (Fix 1)
log_returns <- diff(log(esg_raw))  # ESG log returns, length M-1
```

**Critical alignment fix:** the VIX series has one more observation than the return series. `vix[t]` is the VIX *closing level on the same day* as `r_spx[t]`, which is computed from `spx[t-1]` and `spx[t]`. Therefore `vix[-1]` (dropping the first observation) is passed to the likelihood, ensuring that `vix_obs[t]` and `returns[t]` correspond to the same calendar date.

**Pre-flight assertions** (`stopifnot`) verify:
- `length(vix) == length(r_spx) + 1`
- No `NA` values in any series
- Return and VIX lengths match after alignment

---

### 5.4 Section 3 — Affine Coefficients

```r
vix_affine_coef <- function(omega, alpha, beta, gamma_star, T_H)
```

Runs the forward recursion for `τ = 1, …, T_H` and returns:

- `sumA` = `Σ A(τ)` from `τ = 1` to `T_H`
- `sumB` = `Σ B(τ)` from `τ = 1` to `T_H`
- `phi_Q` = `β + α γ*²` (Q-measure persistence, checked `< 1`)

The key implementation detail is the **order of operations**: propagate first (`A_new`, `B_new`), then accumulate into `sumA`, `sumB`. Accumulating before propagating would shift the sum by one period (the off-by-one error corrected in Fix 4).

---

### 5.5 Section 4 — Joint NLL Function

```r
joint_hn_nll <- function(params, returns, vix_obs, T_H, ann_factor, vix_weight)
```

**Parameter vector** `params[1:6]` = `{ω, α, β, γ, λ, σ_v}`.

The function evaluates the negative joint log-likelihood:

$$\text{NLL} = -\left(\mathcal{L}_{\text{SPX}} + w \cdot \mathcal{L}_{\text{VIX}}\right)$$

**SPX leg** (Gaussian conditional density):

$$\mathcal{L}_{\text{SPX}} = -\frac{1}{2}\sum_t \left[\ln h_t + \frac{(r_t - \lambda h_t)^2}{h_t}\right]$$

**VIX leg** (log-VIX Gaussian measurement equation):

$$\mathcal{L}_{\text{VIX}} = -\frac{1}{2}\sum_t \left[\ln \sigma_v^2 + \frac{(\ln \text{VIX}_t - \frac{1}{2}\ln \widehat{\text{VIX}}_t^2)^2}{\sigma_v^2}\right]$$

The `log(VIX)` formulation is standard in the VIX options literature (Bardgett, Gourier & Leippold 2019) because it maps the positive-valued VIX to the real line and produces residuals that are empirically closer to Gaussian.

**Penalty conditions** returning `1e10`:

| Condition | Reason |
|---|---|
| `ω ≤ 0` or `α ≤ 0` or `β ≤ 0` or `σ_v ≤ 0` | Parameters must be strictly positive |
| `β + αγ² ≥ 1` | Stationarity violation |
| `φ_Q = β + αγ*² ≥ 1` | Q-measure affine recursion diverges |
| `h ≤ 0` or `!is.finite(h)` | Variance collapse |
| `vix2_hat ≤ 0` | Non-positive model VIX² |
| `!is.finite(nll)` | Overflow/underflow |

---

### 5.6 Section 5 — SPX/VIX MLE

Optimisation via `solnp` (Sequential Quadratic Programming, Ye 1987):

```r
opt_result <- solnp(
  pars    = params_init,
  fun     = joint_hn_nll,
  ineqfun = ineq_fun,      # g(θ) = β + αγ²
  ineqLB  = 1e-9,
  ineqUB  = 1 - 1e-6,
  LB      = lb,
  UB      = ub,
  ...
)
```

**Starting values:**

| Parameter | `params_init` | Rationale |
|---|---|---|
| `ω` | `1e-6` | Order of daily variance intercept on decimal returns |
| `α` | `1e-6` | ARCH term tiny in decimal scale |
| `β` | `0.90` | Typical GARCH persistence for equity |
| `γ` | `100` | Large because `√h_t ≈ 0.01`, so `γ√h_t ≈ 1` |
| `λ` | `2.0` | Modest positive variance risk premium |
| `σ_v` | `0.10` | Log-VIX noise ~10% |

**Lower bound rationale (gradient collapse prevention):**

Setting `lb = 0` for `ω` or `α` causes the unconditional variance `h̄ = (ω + α)/(1 - β - αγ²)` to collapse toward zero. At that point `log(h) → -∞` and `∂L/∂ω → ∞`, making the gradient undefined at the boundary. The bounds `lb(ω) = lb(α) = 1e-9` keep the optimiser five orders of magnitude below typical values while imposing no economically meaningful constraint (an annualised volatility floor of `0.016%`).

**Post-estimation derived quantities:**

```r
stat_measure   <- beta_hat + alpha_hat * gamma_hat^2     # stationarity LHS
h_uncond_hat   <- (omega_hat + alpha_hat) / (1 - stat_measure)
gamma_star_hat <- gamma_hat + lambda_hat + 0.5           # risk-neutral leverage
phi_Q          <- beta_hat + alpha_hat * gamma_star_hat^2
```

---

### 5.7 Section 6 — SPX Diagnostic Plots

Four `ggplot2` panels saved to `hn_garch_spx_diagnostics.png`:

| Panel | Content |
|---|---|
| `p1` | Observed vs. model-implied VIX (time series) |
| `p2` | Physical conditional volatility `√(h_t · 252) × 100` |
| `p3` | VIX pricing error with ±RMSE bands |
| `p4` | Scatter: model vs. observed VIX |

---

### 5.8 Section 7 — ESG Physical MLE

**Five-parameter NLL** — `λ_ESG` is estimated freely (not fixed to zero):

```r
nll_esg <- function(params, returns)
# params = {omega, alpha, beta, gamma, lambda_esg}
```

Rationale: fixing `λ = 0` biases the conditional mean equation, which can absorb variation that should be attributed to the variance parameters. Estimating `λ_ESG` freely as a physical drift parameter produces unbiased `{ω, α, β, γ}` estimates. Critically, `λ_ESG` does **not** enter the risk-neutral transformation — that uses `λ_SPX` exclusively.

**Risk-neutral transformation:**

```r
gamma_star_esg <- gamma_esg + lambda_spx + 0.5
```

`lambda_spx` was carried forward from the SPX/VIX joint MLE as `lambda_hat`.

---

### 5.9 Section 8 — ESG P-Measure Filter

```r
for (t in seq_along(log_returns)) {
  h_path_esg[t] <- h_curr
  z_t    <- log_returns[t] / sqrt(h_curr)
  h_curr <- omega_esg + beta_esg * h_curr +
            alpha_esg * (z_t - gamma_esg * sqrt(h_curr))^2
}
```

This produces the **P-measure conditional variance path** `h_path_esg[t]`. The filter uses the physical-measure innovation `z_t = r_t / √h_t` (mean zero because `λ = 0` is a valid approximation for the filter, or the estimated `λ_ESG` can be incorporated without affecting the theoretical validity).

**Key design note:** passing the P-measure filtered `h_t` as `h0` into the Q-measure option pricer is **theoretically correct**. The current variance state is a physical observable — it is the same value regardless of which measure is used for pricing. The measure change only affects the drift and leverage; the current state `h_t` is invariant.

---

### 5.10 Section 9 — CBOE Variance Formula

```r
cboe_variance <- function(K_grid, Q_grid, T, R, F, K0)
```

Exact implementation of the CBOE White Paper formula. The `delta_K` edge treatment uses one-sided differences at the two boundary strikes, matching the CBOE specification precisely. For a uniform grid the interior central differences are constant, but the edge correction is non-trivial and must be coded explicitly.

---

### 5.11 Section 10 — Strike Grid Builder

```r
build_otm_grid <- function(S_t, h_t, T_years, R_annual)
```

**Steps:**

1. Generate `K_grid = S_t × m_grid` where `m_grid` spans `[0.01, 2.01]` uniformly
2. Compute forward `F_t = S_t · exp(R · T)` and find `K0_idx`
3. Call `hn_option_strip(K_grid, ...)` once for puts and once for calls — **vectorised C++ call**, no R loop over strikes
4. Apply OTM selection rule at each strike
5. Apply two-consecutive tail truncation from each wing
6. Return `list(K_grid, Q_grid, F, K0)` with truncated strikes only

**Moneyness grid design:** the domain `[0.01, 2.01]` with 150 strikes gives a constant spacing of `Δm ≈ 0.0134`, corresponding to `ΔK ≈ 0.0134 × S_t`. For `S_t = 100`, this is `ΔK ≈ 1.34` — fine enough to accurately integrate the volatility surface from deep OTM puts to deep OTM calls.

---

### 5.12 Section 11 — Parallelised Daily VIX Loop

The daily VIX computation is **embarrassingly parallel**: each day `t` depends only on `(S_t, h_t)` and the fixed model parameters. The `vix_one_day` function is fully self-contained, embedding its own copies of the grid builder and CBOE formula as inner closures.

**Platform dispatch:**

```r
if (.Platform$OS.type == "windows") {
  cl <- parallel::makeCluster(N_CORES)
  parallel::clusterExport(cl, ...)
  parallel::clusterEvalQ(cl, { library(Rcpp); sourceCpp("hn_pricer.cpp") })
  # parLapply ...
  parallel::stopCluster(cl)
} else {
  # mclapply (fork-based, inherits parent environment)
}
```

**Windows PSOCK cluster:** PSOCK workers are blank R processes with no inherited state. Every object required by `vix_one_day` must be explicitly exported via `clusterExport`. Additionally, each worker must recompile `hn_pricer.cpp` via `clusterEvalQ` because the compiled C++ functions are not serialisable across process boundaries.

**Unix/macOS fork:** `mclapply` uses `fork()` which copies the entire parent memory space. The compiled C++ functions are present in the child processes without any re-export or recompilation.

---

### 5.13 Section 12 — ESG Output

Three output files are written:

**`hn_garch_esg_diagnostics.png`** — two-panel plot:
- ESG synthetic VIX time series
- ESG physical conditional volatility

**`esg_synthetic_vix.csv`** — daily output table:

| Column | Description |
|---|---|
| `Date` | Trading date |
| `VIX_ESG` | GreenVIX level (rounded to 6 d.p.) |
| `Vol_ESG` | Annualised physical volatility `%` |
| `h_t` | Raw conditional variance (10 d.p.) |

**`hn_garch_esg_results.rds`** — full R list containing all parameters and arrays for downstream use.

---

## 6. Parameter Reference

### SPX/VIX Parameters `{ω, α, β, γ, λ, σ_v}`

| Symbol | Name | Typical range (decimal returns) | Interpretation |
|---|---|---|---|
| `ω` | Variance intercept | `1e-7` to `1e-5` | Long-run variance floor |
| `α` | ARCH coefficient | `1e-7` to `1e-5` | Sensitivity to recent shocks |
| `β` | GARCH coefficient | `0.85` to `0.98` | Variance persistence |
| `γ` | Physical leverage | `50` to `300` | Asymmetry; large on decimal scale because `√h_t ≈ 0.01` |
| `λ` | Variance risk premium | `1.0` to `5.0` | Compensation for holding variance risk |
| `σ_v` | VIX noise std dev | `0.05` to `0.30` | Log-VIX measurement error |

### ESG Parameters `{ω, α, β, γ}` + borrowed `λ_SPX`

Same interpretation as above but calibrated to ESG return dynamics. `λ_SPX` is not re-estimated.

### Derived Risk-Neutral Parameters

| Symbol | Formula | Description |
|---|---|---|
| `γ*` | `γ + λ + 0.5` | Q-measure leverage |
| `φ_Q` | `β + α(γ*)²` | Q-measure variance persistence (must be `< 1`) |
| `γ*_ESG` | `γ_ESG + λ_SPX + 0.5` | ESG Q-measure leverage |

---

## 7. Calibration Constraints

| Constraint | Type | Enforcement |
|---|---|---|
| `ω > 0` | Box lower bound | `LB[1] = 1e-9` |
| `α > 0` | Box lower bound | `LB[2] = 1e-9` |
| `β > 0` | Box lower bound | `LB[3] = 1e-6` |
| `β + αγ² < 1` | Inequality | `ineqfun`, `ineqUB = 1 - 1e-6` |
| `σ_v > 0` | Box lower bound | `LB[6] = 1e-6` |
| `φ_Q < 1` | Penalty `1e10` | Inside NLL function |
| `h_t > 0` ∀t | Penalty `1e10` | Inside NLL loop |

The gap `1 - 1e-6` in `ineqUB` prevents the SQP solver's augmented-Lagrangian penalty from evaluating the log-likelihood at a unit-root parameter vector, where `h̄` is infinite.

---

## 8. Data Requirements

- All returns must be in **decimal form** (e.g. `0.012` for 1.2%). Percentage-form data will produce parameters that are wrong by factors of 10,000 and will likely cause the optimiser to fail at the starting values.
- VIX must be in **raw index units** (e.g. `18.5`), not decimal volatility.
- ESG prices must be **adjusted** for dividends and splits. The column `Price (Adjusted BESG)` is expected.
- The three series (SPX, VIX, ESG) need not have identical length but must share a common date range after inner-joining on `Date`.
- No `NA` values are tolerated. Pre-clean the input data.

---

## 9. Outputs

| File | Format | Contents |
|---|---|---|
| `hn_garch_spx_results.rds` | R list | All SPX/VIX MLE parameters + fit statistics |
| `hn_garch_esg_results.rds` | R list | All ESG parameters + VIX series + variance path |
| `esg_synthetic_vix.csv` | CSV | Daily GreenVIX, physical vol, `h_t` |
| `hn_garch_spx_diagnostics.png` | PNG (14×10 in, 180 dpi) | 4-panel SPX fit diagnostic |
| `hn_garch_esg_diagnostics.png` | PNG (14×8 in, 180 dpi) | 2-panel ESG VIX diagnostic |

---

## 10. Design Decisions and Known Limitations

### Why `solnp` and not `optim`?

`optim` with `method = "L-BFGS-B"` cannot handle nonlinear inequality constraints. The stationarity condition `β + αγ² < 1` is nonlinear in the parameters and cannot be expressed as a box constraint. `solnp` implements a full SQP method that handles both box and general inequality constraints simultaneously.

### Why Gil-Pelaez and not a finite-difference PDE?

The HN-GARCH model is affine, meaning the characteristic function is available in closed form. The Gil-Pelaez inversion is exact (up to quadrature error) and requires no spatial grid, no boundary condition tuning, and no time-stepping stability analysis. For the problem at hand (150 strikes × 2 maturities × N days) it is both faster and more accurate than a PDE approach.

### Why borrow `λ_SPX` for the ESG index?

`λ` is the **market price of variance risk** — the compensation investors demand for being exposed to volatility fluctuations. This is a market-wide quantity, not an asset-specific one. An ESG index is a sub-portfolio of the equity market and is exposed to the same aggregate variance risk factor. There is no traded ESG variance market from which to estimate an independent `λ_ESG`, making the transplant methodology the only economically coherent approach.

### Limitations

- The physical-measure filter initialises at the unconditional variance `h̄`. For short samples or at the start of a crisis, this introduces a burn-in bias of approximately 50–100 observations.
- The C++ CF recursion converts `T_years` to trading days by `round(T × 252)`. For maturities that are not multiples of `1/252`, there is a rounding error of up to half a trading day.
- The parallelised loop reprices all 150 strikes × 2 maturities from scratch every day. If `N` is large (> 2,000 days), pre-computing the option surface and caching it will substantially reduce runtime.
- The moneyness grid `[0.01, 2.01]` is fixed. For extreme volatility regimes (e.g. `VIX > 60`), the OTM wings may need to be extended beyond `m = 2.01` to capture tail variance adequately.

---

## 11. References

- Heston, S. L., & Nandi, S. (2000). A closed-form GARCH option valuation model. *Review of Financial Studies*, 13(3), 585–625.
- Bardgett, C., Gourier, E., & Leippold, M. (2019). Inferring volatility dynamics and risk premia from the S&P 500 and VIX markets. *Journal of Financial Economics*, 131(1), 3–26.
- Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38(3–4), 481–482.
- CBOE (2019). *Cboe Volatility Index: VIX White Paper*. Chicago Board Options Exchange.
- Ye, Y. (1987). Interior algorithms for linear, quadratic, and linearly constrained non-linear programming. *Ph.D. Dissertation*, Stanford University. (basis for `Rsolnp`)
- Ghalanos, A., & Theussl, S. (2015). *Rsolnp: General Non-Linear Optimization Using Augmented Lagrange Multiplier Method*. R package.
