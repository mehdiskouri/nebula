"""
Jensen sub-cell-variance correction for nonlinear (Arrhenius) rates (Decision #16;
ARCHITECTURE §III.4 "the nonlinear trap"). Verified by V1.3 (PASS): mean-only lumping
under-estimates a steep-gradient cell's rate by 57-93% and can SILENTLY EXTINGUISH a burn
(consumed 4.6% vs 92.7% of fuel); the variance correction recovers it (<=5.3% for eps<0.5),
and eps is the wired refine trigger.

A homogenized cell carries one mean temperature T-bar, but the Arrhenius rate g(T)=exp(-Ta/T)
is CONVEX over the physical range (Ta/T >> 2), so by Jensen <g(T)> >= g(T-bar): lumping at the
mean systematically UNDER-estimates. The fix carries the sub-cell VARIANCE and applies the
second-order correction g_corr = g(T-bar) + 1/2 g''(T-bar) sigma^2. This is the SECOND tracked
homogenization error -- the variance term that joins V0.1's Voigt-Reuss responses term; the
dimensionless eps = 1/2 sigma^2 |g''/g| is its refine-trigger scalar (refine when eps > eps*~0.5).

Ported verbatim-in-behaviour from the frozen oracle src/verification/oracles/jensen_rate.py
(the fire-coupled extinction demo stays in the oracle; this is the pure rate machinery). Ta is
passed explicitly so this module is independent of any particular operator. Pure numpy.
"""
import numpy as np

EPS = 1e-300


def g(T, Ta):
    """Arrhenius shape g(T) = exp(-Ta/T)."""
    return np.exp(-Ta / np.maximum(T, 1.0))


def g1(T, Ta):
    """g'(T) = g(T) * Ta/T^2  (> 0: rate rises with T)."""
    T = np.maximum(T, 1.0)
    return g(T, Ta) * Ta / T**2


def g2(T, Ta):
    """g''(T) = g(T) * (Ta/T^3)(Ta/T - 2)  (> 0 for Ta/T > 2 -- always, here)."""
    T = np.maximum(T, 1.0)
    return g(T, Ta) * (Ta / T**3) * (Ta / T - 2.0)


def mean_only_rate(Tfield, A, Ta):
    """Naive lumped rate: the Arrhenius law at the cell mean. A * g(T-bar)."""
    return A * g(float(np.mean(Tfield)), Ta)


def true_mean_rate(Tfield, A, Ta):
    """The fine-scale truth: integral of the rate over the resolved sub-cell field. A * <g(T)>."""
    return A * float(np.mean(g(np.asarray(Tfield, float), Ta)))


def variance_corrected_rate(Tfield, A, Ta):
    """Second-order (Jensen) correction: A * (g(T-bar) + 1/2 g''(T-bar) sigma^2)."""
    T = np.asarray(Tfield, float)
    Tbar = float(T.mean()); var = float(T.var())
    return A * (g(Tbar, Ta) + 0.5 * g2(Tbar, Ta) * var)


def mean_only_from_moments(Tbar, A, Ta):
    return A * g(Tbar, Ta)


def corrected_from_moments(Tbar, var, A, Ta):
    return A * (g(Tbar, Ta) + 0.5 * g2(Tbar, Ta) * var)


def variance_error_scalar(Tbar, var, Ta):
    """The dimensionless magnitude of the 2nd-order term, eps = 1/2 sigma^2 |g''(T-bar)/g(T-bar)|.

    The companion of the Voigt-Reuss gap: refine when eps exceeds the validity edge eps* (where
    higher moments overwhelm the 2nd-order truncation, V1.3: eps* ~ 0.5)."""
    return 0.5 * var * abs(g2(Tbar, Ta) / (g(Tbar, Ta) + EPS))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    Ta = 9000.0
    n = 24
    print("1) Jensen sign (cold core 350K, hot face swept) -> mean-only under-estimates:")
    for T_hi in (450, 600, 800, 1000):
        x = (np.arange(n) + 0.5) / n
        f = np.broadcast_to((350.0 + (T_hi - 350.0) * x)[:, None, None], (n, n, n))
        mo = mean_only_rate(f, 3e6, Ta); tr = true_mean_rate(f, 3e6, Ta)
        co = variance_corrected_rate(f, 3e6, Ta)
        eps = variance_error_scalar(f.mean(), f.var(), Ta)
        assert g2(f.mean(), Ta) > 0 and tr >= mo - 1e-30
        print(f"   face={T_hi}K  under-est={100*(1-mo/tr):4.0f}%  corrected err={100*abs(co-tr)/tr:5.1f}%  eps={eps:6.3f}")
    print("\njensen self-checks passed.")
