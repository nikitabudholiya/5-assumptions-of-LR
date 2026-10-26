# 5 Assumptions of Linear Regression

Linear regression works best when 5 conditions hold. If they are violated, the model's predictions or its p-values and confidence intervals can be misleading.

| # | Assumption | One-line meaning |
|---|------------|------------------|
| 1 | Linearity | Input and output have a straight-line relationship |
| 2 | No multicollinearity | Input columns should not depend on each other |
| 3 | Normality of residuals | Residuals should follow a normal (bell-shaped) distribution |
| 4 | Homoscedasticity | Residuals should have the same spread everywhere |
| 5 | No autocorrelation | Residuals should not be related to each other |

**Residual = actual value - predicted value**

---

## 1. Linearity

The relationship between input (X) and output (y) should be a straight line.

![Linearity](01_linearity.png)

- **Check:** scatter plot of X vs y, or residuals vs predicted values (there should be no curve).
- **Fix:** transform the features (log, square root), add polynomial terms, or use a non-linear model.

---

## 2. No Multicollinearity

Input columns should not be highly correlated with each other. If one feature can be predicted from another, the model cannot tell which one is actually driving the output, so the coefficients become unstable.

![Multicollinearity](02_multicollinearity.png)

- **Check:** correlation matrix, or VIF (Variance Inflation Factor). VIF above 5 to 10 is a warning sign.
- **Fix:** drop one of the correlated features, combine them, or use PCA / Ridge regression.

---

## 3. Normality of Residuals

If we plot the distribution of the residuals, it should look like a bell curve centered at 0.

![Normality](03_normality.png)

- **Check:** histogram of residuals, Q-Q plot, or Shapiro-Wilk test.
- **Fix:** transform the target (log, Box-Cox), or remove extreme outliers.
- **Note:** this matters mainly for confidence intervals and p-values. With large datasets it is less critical.

---

## 4. Homoscedasticity

The residuals should have the same scatter at every predicted value. When the spread grows or shrinks (a funnel shape), it is called heteroscedasticity.

![Homoscedasticity](04_homoscedasticity.png)

- **Check:** plot residuals vs predicted values, or run the Breusch-Pagan test.
- **Fix:** transform the target (log), or use weighted least squares.

---

## 5. No Autocorrelation

Residuals should be independent of each other. One residual should not tell us anything about the next one. This is mostly a problem in time-series data.

![Autocorrelation](05_autocorrelation.png)

- **Check:** plot residuals in order, or use the Durbin-Watson test (a value near 2 means no autocorrelation).
- **Fix:** add lag features, or use time-series models like ARIMA.

---

## Python: Quick Checks

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

X_const = sm.add_constant(X)           # X: DataFrame of features
model = sm.OLS(y, X_const).fit()
residuals = model.resid

# 2. Multicollinearity (VIF per feature)
vif = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])]

# 3. Normality of residuals
print(stats.shapiro(residuals))

# 4. Homoscedasticity
print(het_breuschpagan(residuals, X_const))

# 5. Autocorrelation (close to 2 is good)
print(durbin_watson(residuals))
```

---

## Quick Summary

> **L**inear relationship, **I**ndependent features, **N**ormal residuals, **E**qual variance, **A**ll residuals independent.

Remember it as **LINEA**.
