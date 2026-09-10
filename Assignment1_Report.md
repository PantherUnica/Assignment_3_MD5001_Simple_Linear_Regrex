# Assignment 1 — Simple Linear Regression: TV Advertising vs. Sales

**Author:** Aditya Raj
**Dataset:** `tv_sales_data.csv` (TV budget in $1,000s, Sales in 1,000s of units, n = 6)
**Model:** Sales = β0 + β1 · TV + ε
**Method:** Coefficients fit via **Gradient Descent**; standard errors, R², and p-values computed
via the exact simple-linear-regression inference formulas applied to the fit's own residuals.

---

## 1. Code and Method

The model is fit by **Gradient Descent** instead of a closed-form formula: it starts both
coefficients at 0, then repeatedly nudges them in the direction that reduces the Mean Squared
Error (MSE), using a learning rate of `α = 0.1` for `1000` iterations. The TV values are
standardized before the loop purely for numerically stable steps, and the learned coefficients
are converted back to the original ($1,000s) TV scale afterward.

```python
x = df["TV"].values.astype(float)
y_vals = df["Sales"].values.astype(float)
n = len(x)

x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std

b0, b1 = 0.0, 0.0
alpha = 0.1
n_iters = 1000

for i in range(n_iters):
    y_pred = b0 + b1 * x_scaled
    error = y_pred - y_vals
    grad_b0 = (2 / n) * np.sum(error)
    grad_b1 = (2 / n) * np.sum(error * x_scaled)
    b0 -= alpha * grad_b0
    b1 -= alpha * grad_b1

beta1 = b1 / x_std
beta0 = b0 - beta1 * x_mean
```

Gradient Descent only minimizes MSE — it does not, by itself, produce standard errors,
R², or p-values. Those are calculated **directly from the fitted line's own residuals**, using
the exact simple-linear-regression inference formulas:

```python
from scipy import stats

y_fit = beta0 + beta1 * x
residuals = y_vals - y_fit

RSS = np.sum(residuals ** 2)
TSS = np.sum((y_vals - y_vals.mean()) ** 2)
df_resid = n - 2
sigma2 = RSS / df_resid
Sxx = np.sum((x - x_mean) ** 2)

se_beta0 = np.sqrt(sigma2 * (1 / n + x_mean ** 2 / Sxx))
se_beta1 = np.sqrt(sigma2 / Sxx)

t_beta0 = beta0 / se_beta0
t_beta1 = beta1 / se_beta1
p_beta0 = 2 * (1 - stats.t.cdf(abs(t_beta0), df_resid))
p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), df_resid))

r_squared = 1 - RSS / TSS
```

The formulas used:

| Quantity | Formula |
|---|---|
| Residual variance | σ̂² = RSS / (n − 2) |
| SE(β1) | √( σ̂² / Sxx ),  Sxx = Σ(xi − x̄)² |
| SE(β0) | √( σ̂² · (1/n + x̄²/Sxx) ) |
| t-statistic | β̂ / SE(β̂), compared to a t-distribution with n − 2 degrees of freedom |
| R² | 1 − RSS/TSS |

---

## 2. Results

| Quantity | Value |
|---|---|
| β0 (intercept) | **3.6000** |
| β1 (slope, TV) | **0.5657** |
| SE(β0) | 0.5674 |
| SE(β1) | 0.0291 |
| t-statistic (β0) | 6.345 |
| t-statistic (β1) | 19.415 |
| p-value (intercept) | 0.003160 |
| p-value (slope) | 4.15 × 10⁻⁵ |
| R-squared | **0.9895** |
| Degrees of freedom | 4 (n = 6, 2 parameters) |

Fitted equation: **Sales = 3.60 + 0.5657 × TV**

### Plots

**Fit and residuals:**

![Regression fit and residuals](regression_plot.png)

**Gradient Descent cost convergence:**

![Gradient Descent cost convergence](gradient_descent_plot.png)

The MSE falls smoothly toward its minimum over 1,000 iterations, confirming the model
converged to a stable fit rather than stopping early or diverging.

---

## 3. Interpretation — Plain Language

- **What β1 means in everyday terms:** β1 = 0.5657 is a positive number, so spending more on
  TV ads goes with more sales. Since TV is measured in $1,000s and Sales in 1,000s of units,
  this means: for every extra **$1,000** put into TV advertising, sales go up by roughly
  **566 units**, on average.
- **Is this a real effect, or could it be luck?** The p-value for the slope is
  0.0000415 — a tiny number, way below the usual 0.05 cutoff people use to decide if
  something is "statistically significant." In plain words: it's extremely unlikely we'd see
  a relationship this strong just by random chance if TV spend actually had no effect on
  sales. So yes, there's strong evidence TV advertising really does move sales.
- **How much of sales is "explained" by TV budget?** R² = 0.9895, meaning about **99%** of
  the ups and downs in sales across these 6 data points can be accounted for just by knowing
  the TV budget. Only about 1% is left over as unexplained noise. That's an extremely tight
  fit — expected here because this is a small, clean practice dataset (only 6 points), not
  noisy real-world data.
- **Bottom line:** the data says TV advertising budget is a strong, reliable, positive driver
  of sales in this dataset, and a simple straight-line model captures almost all of the
  pattern.

---

## 4. Interpretation — Professional / Technical Language

- **Sign and magnitude of β1:** The estimated slope, β̂1 = 0.5657, is positive and indicates
  a positive linear association between TV advertising expenditure and sales volume. Holding
  the linear specification fixed, each additional $1,000 unit of TV budget is associated with
  an expected increase of approximately 0.5657 thousand units (≈566 units) in sales.
- **Statistical significance:** Under H0: β1 = 0 versus H1: β1 ≠ 0, the test statistic
  t = β̂1 / SE(β̂1) = 19.415 on n − 2 = 4 degrees of freedom yields p ≈ 4.15 × 10⁻⁵, which is
  far below conventional significance thresholds (α = 0.05, 0.01). H0 is rejected, providing
  strong statistical evidence of a non-zero linear relationship between TV advertising and
  sales. The intercept is likewise statistically significant (β̂0 = 3.60, SE = 0.5674,
  t = 6.345, p ≈ 0.0032), though its interpretation as "expected sales at zero TV spend" is a
  mathematical extrapolation rather than a directly observed operating condition, since the
  sample range of TV spend is $5,000–$30,000.
- **Proportion of variance explained:** R² = 0.9895 indicates that approximately 98.95% of
  the total variability in sales (TSS) is explained by the fitted linear relationship with TV
  advertising budget, leaving only ≈1.05% attributable to residual (unexplained) variation.
  Such a high R² is atypical for real marketing data and reflects the small sample size
  (n = 6) and the near-linear, low-noise construction of this illustrative dataset rather than
  a general claim about TV advertising's explanatory power in practice.
- **Overall conclusion:** Within this sample, TV advertising budget is a statistically
  significant and practically strong positive predictor of sales, and the simple linear model
  accounts for nearly all of the observed variance in the response.
