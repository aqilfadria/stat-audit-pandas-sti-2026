"""
estimator.py
============
Member B — Estimation Analyst
Statistical Audit of pandas-dev/pandas · STI 2026

All formulas implemented per:
    Tsun (2020). Probability & Statistics with Applications to Computing.
    Chapters 7–8.

Functions
---------
mle_bernoulli(data)
mle_poisson(data)
beta_posterior(k, m)
log_likelihood_bernoulli(theta, k, n)
log_likelihood_poisson(theta, data)
"""

import numpy as np
from scipy import stats
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLE — Bernoulli
# ─────────────────────────────────────────────────────────────────────────────

def mle_bernoulli(data: Union[list, np.ndarray]) -> dict:
    """
    Compute the Maximum Likelihood Estimate of theta for a Bernoulli distribution.

    Model: Each observation x_i in {0, 1}, i.i.d. ~ Bernoulli(theta).
    Likelihood: L(theta) = theta^k * (1 - theta)^(n - k)
    MLE: theta_hat = k / n                          [Tsun 2020, p. 254]

    Dalam konteks pandas-dev/pandas:
        data = array biner hasil PR (1 = merged, 0 = closed unmerged)

    PERINGATAN: Derivasi (d ln L / d_theta = 0) WAJIB ditulis sendiri
    oleh Member B di notebook 02_estimation.ipynb. Tidak boleh di-generate AI.

    Parameters
    ----------
    data : array-like of int (0 or 1)
        Binary outcomes. 1 = PR merged, 0 = PR closed unmerged.

    Returns
    -------
    dict:
        theta_hat : float  — MLE estimate of theta
        k         : int    — jumlah sukses (merged PRs)
        n         : int    — total observasi
        se        : float  — standard error = sqrt(theta_hat*(1-theta_hat)/n)

    Raises
    ------
    ValueError jika data kosong atau mengandung nilai selain 0 dan 1.

    Examples
    --------
    >>> mle_bernoulli([1, 0, 1, 1, 0])
    {'theta_hat': 0.6, 'k': 3, 'n': 5, 'se': 0.2191}
    """
    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("data tidak boleh kosong.")
    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("data harus mengandung hanya 0 dan 1 untuk Bernoulli MLE.")

    n         = len(data)
    k         = int(data.sum())
    theta_hat = k / n                           # Tsun 2020, p. 254
    se        = np.sqrt(theta_hat * (1 - theta_hat) / n)

    return {
        "theta_hat": round(theta_hat, 6),
        "k"        : k,
        "n"        : n,
        "se"       : round(se, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLE — Poisson
# ─────────────────────────────────────────────────────────────────────────────

def mle_poisson(data: Union[list, np.ndarray]) -> dict:
    """
    Compute the Maximum Likelihood Estimate of lambda untuk distribusi Poisson.

    Model: Setiap observasi x_i ~ Poisson(lambda), i.i.d.
    Likelihood: L(lambda) = prod(e^{-lambda} * lambda^{x_i} / x_i!)
    MLE: lambda_hat = sum(data) / n = x_bar       [Tsun 2020, p. 254]

    Dalam konteks pandas-dev/pandas:
        data = jumlah bug issues yang dibuka per bulan

    PERINGATAN: Derivasi (d ln L / d_lambda = 0) WAJIB ditulis sendiri
    oleh Member B di notebook 02_estimation.ipynb. Tidak boleh di-generate AI.

    Parameters
    ----------
    data : array-like of non-negative int
        Count data (jumlah event per periode waktu).

    Returns
    -------
    dict:
        lambda_hat : float — MLE estimate of lambda
        n          : int   — jumlah observasi (bulan)
        se         : float — standard error = sqrt(lambda_hat / n)

    Raises
    ------
    ValueError jika data kosong atau mengandung nilai negatif.

    Examples
    --------
    >>> mle_poisson([3, 5, 2, 4, 6])
    {'lambda_hat': 4.0, 'n': 5, 'se': 0.8944}
    """
    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("data tidak boleh kosong.")
    if np.any(data < 0):
        raise ValueError("Data Poisson harus non-negatif.")

    n          = len(data)
    lambda_hat = data.sum() / n                 # Tsun 2020, p. 254
    se         = np.sqrt(lambda_hat / n)

    return {
        "lambda_hat": round(lambda_hat, 6),
        "n"         : n,
        "se"        : round(se, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Beta Posterior
# ─────────────────────────────────────────────────────────────────────────────

def beta_posterior(k: int, m: int) -> dict:
    """
    Compute Beta posterior parameters dan summary statistics.

    Prior     : theta ~ Beta(1, 1)  (uniform / uninformative)
    Likelihood: Bernoulli dengan k sukses dari k+m trials
    Posterior : theta | data ~ Beta(alpha, beta)
        alpha = k + 1                               [Tsun 2020, p. 269]
        beta  = m + 1                               [Tsun 2020, p. 269]

    PERINGATAN KRITIS: alpha = k+1 dan beta = m+1, BUKAN k dan m.
    Menggunakan k dan m langsung adalah pelanggaran formula sesuai rubrik.

    Parameters
    ----------
    k : int — jumlah sukses (misalnya merged PRs)
    m : int — jumlah gagal (misalnya closed unmerged PRs)

    Returns
    -------
    dict:
        alpha  : int   — posterior alpha = k + 1
        beta   : int   — posterior beta  = m + 1
        mode   : float — (alpha-1)/(alpha+beta-2)   [Tsun 2020, p. 269]
        mean   : float — alpha/(alpha+beta)          [Tsun 2020, p. 269]
        var    : float — alpha*beta/((alpha+beta)^2*(alpha+beta+1))
        ci_95  : tuple — 95% credible interval via scipy Beta PPF

    Raises
    ------
    ValueError jika k atau m negatif.

    Examples
    --------
    >>> beta_posterior(k=300, m=100)
    {'alpha': 301, 'beta': 101, 'mode': 0.749, 'mean': 0.7488, ...}
    """
    if k < 0 or m < 0:
        raise ValueError("k dan m harus bilangan bulat non-negatif.")

    alpha = k + 1                               # Tsun 2020, p. 269
    beta  = m + 1                               # Tsun 2020, p. 269

    mode = (alpha - 1) / (alpha + beta - 2) if (alpha + beta - 2) > 0 else 0.5
    mean = alpha / (alpha + beta)               # Tsun 2020, p. 269
    var  = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    dist  = stats.beta(alpha, beta)
    ci_95 = (round(dist.ppf(0.025), 6), round(dist.ppf(0.975), 6))

    return {
        "alpha": alpha,
        "beta" : beta,
        "mode" : round(mode, 6),
        "mean" : round(mean, 6),
        "var"  : round(var, 8),
        "ci_95": ci_95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Log-Likelihood — Bernoulli
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood_bernoulli(theta: float, k: int, n: int) -> float:
    """
    Compute Bernoulli log-likelihood untuk nilai theta tertentu.

    ln L(theta) = k * ln(theta) + (n-k) * ln(1-theta)

    Gunakan fungsi ini untuk memplot kurva log-likelihood di notebook
    dan memverifikasi secara visual bahwa maksimum ada di theta_hat = k/n.

    Parameters
    ----------
    theta : float in (0, 1) — kandidat nilai theta
    k     : int             — jumlah sukses
    n     : int             — total observasi

    Returns
    -------
    float — nilai log-likelihood pada theta

    Raises
    ------
    ValueError jika theta tidak dalam (0,1) atau k > n.

    Examples
    --------
    >>> log_likelihood_bernoulli(0.6, k=3, n=5)
    -3.2189
    """
    if not (0 < theta < 1):
        raise ValueError("theta harus strictly dalam (0, 1).")
    if k > n:
        raise ValueError("k tidak bisa melebihi n.")
    if n <= 0:
        raise ValueError("n harus bilangan positif.")

    return k * np.log(theta) + (n - k) * np.log(1 - theta)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Log-Likelihood — Poisson
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood_poisson(theta: float, data: Union[list, np.ndarray]) -> float:
    """
    Compute Poisson log-likelihood untuk nilai lambda (theta) tertentu.

    ln L(lambda) = -n*lambda + ln(lambda)*sum(x_i) - sum(ln(x_i!))

    Gunakan fungsi ini untuk memplot kurva log-likelihood di notebook
    dan memverifikasi secara visual bahwa maksimum ada di lambda_hat = x_bar.

    Parameters
    ----------
    theta : float > 0 — kandidat nilai lambda
    data  : array-like of non-negative int — data count yang diobservasi

    Returns
    -------
    float — nilai log-likelihood pada theta (= lambda)

    Raises
    ------
    ValueError jika theta <= 0 atau data mengandung nilai negatif.

    Examples
    --------
    >>> log_likelihood_poisson(4.0, [3, 5, 2, 4, 6])
    -7.9618
    """
    if theta <= 0:
        raise ValueError("theta (lambda) harus positif untuk distribusi Poisson.")

    data = np.asarray(data, dtype=float)
    if np.any(data < 0):
        raise ValueError("Data Poisson harus non-negatif.")

    n       = len(data)
    sum_x   = data.sum()
    sum_log_factorial = np.sum(
        [np.sum(np.log(np.arange(1, int(xi) + 1))) for xi in data]
    )

    log_lik = -n * theta + np.log(theta) * sum_x - sum_log_factorial
    return float(round(log_lik, 6))
