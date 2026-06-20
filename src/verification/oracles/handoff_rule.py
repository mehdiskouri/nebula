"""
RVE <-> surrogate handoff decision rule + uncertainty calibration (V2.1, Risk #1).

In the violent regime the Voigt-Reuss analytic bound is INVALID (dns_damage_3d shows the true
response leaving the bracket), so the per-cell trust scalar has no cheap closed form. The only
remaining signal is the surrogate's SELF-UNCERTAINTY u. V2.1 asks two things, staged:

  (1) CALIBRATION — is u a faithful estimate of the surrogate's actual error? (rank correlation;
      not systematically over-confident). If this fails, the rule stage is moot and the protocol
      fallback is CONSTRAIN-hard: always RVE-solve in the violent regime.
  (2) RULE — the policy "u > tau -> pay for an exact RVE solve, else trust the surrogate" traces a
      frontier between STALLING (always solve: RVE-fraction -> 1) and LYING (always trust: tail
      error large). A good operating point keeps tail outcome error below a bound AND the RVE
      fraction below a cost budget — a genuine interior tradeoff.

Pure numpy/scipy; the surrogate, its uncertainty, and the RVE ground truth come from
surrogate_gnn / dns_damage_3d. This module is just the decision calculus over those numbers.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


# ------------------------------------------------------------------- calibration
def rank_correlation(u, abs_err):
    """Raw Spearman rank correlation between predicted uncertainty and actual |error|.

    NOTE: even a perfectly-calibrated u rank-correlates only weakly with a *single-draw* |error|
    (the error magnitude is itself random given u). The systematic-component measure
    `binned_rank_correlation` is the right reliability test; this raw value is reported as context.
    """
    rho, _ = spearmanr(u, abs_err)
    return float(rho)


def binned_rank_correlation(u, abs_err, n_bins=10):
    """Reliability correlation: bin cells by u, correlate (mean u) vs (RMS error) across bins.

    Averaging within bins removes the single-draw noise and exposes whether u tracks the
    SYSTEMATIC error level — the quantity the protocol's '>0.8 rank correlation' refers to.
    """
    u = np.asarray(u, float); abs_err = np.asarray(abs_err, float)
    order = np.argsort(u)
    bins = np.array_split(order, min(n_bins, len(u)))
    mu = np.array([u[b].mean() for b in bins if len(b)])
    rmse = np.array([np.sqrt((abs_err[b] ** 2).mean()) for b in bins if len(b)])
    rho, _ = spearmanr(mu, rmse)
    return float(rho)


def coverage(abs_err, u, ks=(1.0, 2.0)):
    """Observed coverage at k*sigma vs the Gaussian-nominal (68.3%, 95.5%).

    Returns dict k -> (observed, nominal). observed < nominal == OVER-confident (u too small).
    """
    from math import erf, sqrt
    out = {}
    for k in ks:
        nominal = erf(k / sqrt(2.0))
        observed = float(np.mean(abs_err <= k * u))
        out[k] = (observed, nominal)
    return out


def overconfidence(abs_err, u, k=1.0):
    """Nominal-minus-observed coverage at k*sigma; > 0 == systematically over-confident."""
    obs, nom = coverage(abs_err, u, ks=(k,))[k]
    return float(nom - obs)


# ------------------------------------------------------------------- decision rule
@dataclass
class Frontier:
    taus: np.ndarray
    rve_frac: np.ndarray      # fraction of cells sent to the exact RVE solve (the cost)
    tail_err: np.ndarray      # P95 of the POLICY outcome error (0 on RVE'd cells)
    mean_err: np.ndarray


def policy_error(u, rel_err, tau):
    """Per-cell outcome error of the policy at threshold tau (RVE'd cells are exact -> 0)."""
    trust = u <= tau
    return np.where(trust, rel_err, 0.0)


def frontier(u, rel_err, gate=None, taus=None, tail_q=0.95):
    """Sweep tau; for each report RVE-fraction (cost) and the policy's tail / mean outcome error.

    Extremes are the two named failure modes: tau -> inf is ALWAYS-TRUST (rve_frac 0, tail = the
    surrogate's own tail error = LYING); tau -> 0 is ALWAYS-RVE (rve_frac 1, tail 0 = STALLING).

    `gate` (bool mask) = cells the multi-signal trigger sends to RVE unconditionally (envelope-exit
    OR percolation), independent of u. Used to model the architecture's multi-signal fallback.
    """
    u = np.asarray(u, float); rel_err = np.asarray(rel_err, float)
    gate = np.zeros(len(u), bool) if gate is None else np.asarray(gate, bool)
    if taus is None:
        fu = u[~gate]
        lo = (fu.min() * 0.5) if fu.size else 0.0
        hi = (fu.max() * 1.5 + 1e-12) if fu.size else 1.0
        taus = np.concatenate([[0.0], np.linspace(lo, hi, 60), [np.inf]])
    rf, te, me = [], [], []
    for tau in taus:
        rve = gate | (u > tau)                       # RVE if gated OR uncertain
        pe = np.where(rve, 0.0, rel_err)
        rf.append(float(np.mean(rve)))
        te.append(float(np.quantile(pe, tail_q)))
        me.append(float(pe.mean()))
    return Frontier(np.asarray(taus), np.asarray(rf), np.asarray(te), np.asarray(me))


def operating_point(fr: Frontier, err_bound, rve_budget):
    """Best (lowest-cost) tau whose tail error < err_bound. Report if it also fits the RVE budget.

    Returns dict: exists (tail bound met at all), tau, rve_frac, tail_err, within_budget.
    """
    ok = np.where(fr.tail_err < err_bound)[0]
    if ok.size == 0:
        return dict(exists=False, within_budget=False, tau=None,
                    rve_frac=1.0, tail_err=float(fr.tail_err.min()))
    j = ok[np.argmin(fr.rve_frac[ok])]            # cheapest threshold meeting the error bound
    return dict(exists=True, tau=float(fr.taus[j]), rve_frac=float(fr.rve_frac[j]),
                tail_err=float(fr.tail_err[j]), within_budget=bool(fr.rve_frac[j] < rve_budget))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(0)
    N = 400

    # 1) WELL-CALIBRATED synthetic, skewed difficulty: most cells easy (small u), a hard minority
    #    (large u) — the realistic violent-regime population where a cheap interior point exists.
    hard = rng.random(N) < 0.18
    u = np.where(hard, rng.uniform(0.15, 0.35, N), rng.uniform(0.005, 0.04, N))
    rel_err = np.abs(rng.normal(0.0, u))                      # error ~ N(0, u) -> u predicts |err|
    rho = rank_correlation(u, rel_err)
    rho_b = binned_rank_correlation(u, rel_err)
    oc = overconfidence(rel_err, u, k=1.0)
    print(f"1) calibrated: raw spearman = {rho:.3f}; binned reliability rho = {rho_b:.3f}; "
          f"over-confidence@1sigma = {oc:+.3f}")
    cov = coverage(rel_err, u)
    for k, (o, n) in cov.items():
        print(f"     coverage @{k:.0f}sigma: observed={o:.3f} nominal={n:.3f}")
    assert rho_b > 0.8, "well-calibrated u must track the systematic error level (binned rho)"
    assert abs(oc) < 0.1, "should not be systematically over/under-confident"

    fr = frontier(u, rel_err)
    op = operating_point(fr, err_bound=0.10, rve_budget=0.30)
    print(f"2) rule: operating point exists={op['exists']} tau={op['tau']:.3f} "
          f"rve_frac={op['rve_frac']:.3f} tail_err={op['tail_err']:.3f} "
          f"within_budget={op['within_budget']}")
    # extremes confirm the two failure modes
    always_trust_tail = fr.tail_err[-1]          # tau=inf
    always_rve_frac = fr.rve_frac[0]             # tau=0
    print(f"   always-trust tail err (LYING) = {always_trust_tail:.3f}; "
          f"always-RVE fraction (STALLING) = {always_rve_frac:.3f}")
    assert op["exists"] and op["within_budget"], "a viable interior operating point must exist"
    assert always_rve_frac == 1.0, "tau=0 must send everything to RVE"
    assert 0.0 < op["rve_frac"] < 1.0, "operating point must be a genuine interior tradeoff"

    # 2) UNCALIBRATED control: u is pure noise, independent of error -> rho ~ 0 (rule is moot).
    u2 = rng.uniform(0.01, 0.30, N)
    rho2 = binned_rank_correlation(u2, rel_err)
    print(f"3) uncalibrated control: binned rho = {rho2:.3f} (must be << 0.8)")
    assert abs(rho2) < 0.6, "independent u must not track error — confirms the test has teeth"
    print("\nALL handoff_rule self-checks PASSED")
