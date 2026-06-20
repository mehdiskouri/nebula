"""
Physics-informed graph-net surrogate with calibrated uncertainty (V2.4, Decision #17).

The learned tier that runs "on the same hypergraph": a small message-passing network over a
cell's coarse region-adjacency graph (`violent_cells.region_graph`), CONDITIONED on the
homogenized descriptor (`violent_cells.descriptor`) — the same vector the restriction
operator emits. It predicts the violent-regime outcome (degraded stiffness RETENTION
r = C_secant[d,d] / C0[d,d] in (0,1]) the damage-DNS oracle produces, together with a
self-uncertainty that V2.4 needs for OOD detection and V2.1 calibrates against truth.

Three pieces the protocol demands:
  - PHYSICS-INFORMED loss: data NLL + a governing-trend residual (retention must stay in
    (0,1] and must not increase with material contrast — autodiff penalty on d mu / d log-
    contrast) + bound consistency. Buys data efficiency vs a pure-data baseline (V2.4 metric 3).
  - UNCERTAINTY: a deep ensemble (epistemic, variance of member means) of heteroscedastic
    heads (aleatoric, predicted log-variance). u = sqrt(epistemic + aleatoric).
  - OOD DETECTION: u rises off the training manifold; a threshold calibrated on in-family
    quantiles flags OOD cells (V2.4 metric 2).

Fixed region count R -> every cell is the same graph size, so a batch is a dense tensor with
one shared edge_index (no torch_geometric needed). torch; runs on GPU when available. Weights
are saved/loaded so notebooks re-run reproducibly (V0.5 determinism discipline).
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import violent_cells as vc
import dns_damage_3d as dd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGCONTRAST_CH = 19          # descriptor channel holding log10(contrast) — a monotone axis
SOFTFRAC_CH = 18             # descriptor channel holding soft-phase fraction — a monotone axis

# Frozen damage-path parameters for the whole V2.4/V2.1 dataset: this load regime yields a
# GRADED violent-regime outcome (end-stiffness collapses to ~0 for every cell — a useless
# binary — but peak strength / dissipation vary smoothly with geometry).
DATA_PARAMS = dd.DamageParams(n_increments=16, max_strain=3.0e-3, k0=1.2e-3, kf=5e-3)


def set_determinism(seed):
    """Seed all RNGs and request deterministic kernels (declared-tolerance regime, V0.5)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- featurization
def outcome_target(res, k0, load_dir=0):
    """Headline violent-regime outcome: NORMALIZED PEAK STRENGTH y = peak_stress / (C0*k0).

    C0*k0 is the cheap linear/elastic onset-stress estimate (the proxy that ignores
    localization and is WRONG in the violent regime); y in (0,1] is the fraction actually
    achieved — graded, monotone-decreasing in depth and contrast, and exactly the post-yield
    quantity the Voigt-Reuss bound cannot touch.
    """
    sigma_el = res.C0_linear[load_dir, load_dir] * k0
    return float(res.peak_stress / sigma_el)


def build_dataset(samples, params=DATA_PARAMS, cache=None):
    """Run damage-DNS over `samples`; return arrays {y, peak_stress, dissipation, ret_end}.

    The DNS solves are the expensive part — cache to `.npz` so notebooks re-run cheaply. Samples
    are regenerated deterministically from a seed by the caller, so the cache is keyed only by
    name; this asserts the cached length matches.
    """
    if cache is not None and os.path.exists(cache):
        d = np.load(cache)
        assert len(d["y"]) == len(samples), f"cache {cache} length mismatch — delete to rebuild"
        return {k: d[k] for k in d.files}
    y, peak, diss, ret = [], [], [], []
    for s in samples:
        r = dd.run_path(s.grid, s.materials, params)
        y.append(outcome_target(r, params.k0))
        peak.append(r.peak_stress)
        diss.append(r.dissipated_energy)
        ret.append(float(r.C_secant_final[0, 0] / r.C0_linear[0, 0]))
    out = {k: np.array(v) for k, v in
           dict(y=y, peak_stress=peak, dissipation=diss, ret_end=ret).items()}
    if cache is not None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **out)
    return out


def featurize(samples, R=4):
    """Samples -> (node_feats (B,N,2), edge_index (2,E), descriptors (B,20)). Shared topology."""
    nfeats, descs = [], []
    edge_index = None
    for s in samples:
        nf, ei = vc.region_graph(s.grid, s.materials, R=R)
        nfeats.append(nf)
        descs.append(vc.descriptor(s.grid, s.materials))
        edge_index = ei
    X_nodes = torch.tensor(np.stack(nfeats), dtype=torch.float64, device=DEVICE)
    X_desc = torch.tensor(np.stack(descs), dtype=torch.float64, device=DEVICE)
    E = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    return X_nodes, E, X_desc


@dataclass
class Normalizer:
    """Standardize descriptors (fit on training data; reused for all evaluation)."""
    mu: torch.Tensor
    sd: torch.Tensor

    @staticmethod
    def fit(X_desc):
        return Normalizer(mu=X_desc.mean(0), sd=X_desc.std(0) + 1e-9)

    def __call__(self, X_desc):
        return (X_desc - self.mu) / self.sd


# ------------------------------------------------------------------- model
class GraphPIGN(nn.Module):
    """Message-passing net over the region graph, conditioned on the homogenized descriptor."""

    def __init__(self, node_dim=2, desc_dim=20, H=32, n_layers=3):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, H), nn.SiLU())
        self.desc_enc = nn.Sequential(nn.Linear(desc_dim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.SiLU()) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * H, H), nn.SiLU()) for _ in range(n_layers)])
        self.head_mu = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))
        self.head_logvar = nn.Sequential(nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))
        self.double()

    def forward(self, X_nodes, edge_index, X_desc_norm):
        B, N, _ = X_nodes.shape
        g = self.desc_enc(X_desc_norm)                      # (B,H)
        h = self.node_enc(X_nodes)                          # (B,N,H)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp_(min=1.0)
        for msg, upd in zip(self.msg, self.upd):
            g_e = g[:, None, :].expand(B, src.shape[0], -1)             # (B,E,H)
            m = msg(torch.cat([h[:, src, :], h[:, dst, :], g_e], dim=-1))  # (B,E,H)
            agg = torch.zeros_like(h).index_add_(1, dst, m) / deg[None, :, None]
            h = h + upd(torch.cat([h, agg], dim=-1))
        pooled = h.mean(dim=1)                              # (B,H)
        z = torch.cat([pooled, g], dim=-1)                 # (B,2H)
        return self.head_mu(z).squeeze(-1), self.head_logvar(z).squeeze(-1)


# ------------------------------------------------------------------- training
@dataclass
class TrainCfg:
    H: int = 32
    n_layers: int = 3
    epochs: int = 400
    lr: float = 3e-3
    physics: bool = True
    w_range: float = 1.0          # weight on the (0,1] range penalty
    w_mono: float = 3.0           # weight on the monotone-strength penalties


def _physics_penalty(model, X_nodes, edge_index, X_desc_norm, cfg):
    """Governing-trend residual: strength in (0,1] AND non-increasing in BOTH contrast and the
    soft-phase fraction (more/softer char -> weaker). These monotonicities are physical priors
    the smooth data only learns slowly, so they buy low-data accuracy (the PINN claim)."""
    Xd = X_desc_norm.clone().requires_grad_(True)
    mu, _ = model(X_nodes, edge_index, Xd)
    grad = torch.autograd.grad(mu.sum(), Xd, create_graph=True)[0]      # d mu / d desc
    mono = (torch.relu(grad[:, LOGCONTRAST_CH]).mean()
            + torch.relu(grad[:, SOFTFRAC_CH]).mean())                 # penalize increasing
    rng = (torch.relu(mu - 1.0) ** 2 + torch.relu(-mu + 1e-3) ** 2).mean()
    return cfg.w_range * rng + cfg.w_mono * mono


def train_member(samples, y, cfg: TrainCfg, seed, normalizer=None):
    """Train one heteroscedastic GraphPIGN member; returns (model, normalizer)."""
    set_determinism(seed)
    X_nodes, E, X_desc = featurize(samples)
    if normalizer is None:
        normalizer = Normalizer.fit(X_desc)
    Xd = normalizer(X_desc)
    yt = torch.tensor(np.asarray(y), dtype=torch.float64, device=DEVICE)
    model = GraphPIGN(desc_dim=X_desc.shape[1], H=cfg.H, n_layers=cfg.n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        mu, logvar = model(X_nodes, E, Xd)
        logvar = logvar.clamp(-8.0, 4.0)
        nll = 0.5 * (torch.exp(-logvar) * (yt - mu) ** 2 + logvar).mean()
        loss = nll + (_physics_penalty(model, X_nodes, E, Xd, cfg) if cfg.physics else 0.0)
        loss.backward()
        opt.step()
    return model, normalizer


class Ensemble:
    """Deep ensemble of heteroscedastic members -> mean + epistemic/aleatoric uncertainty."""

    def __init__(self, members, normalizer):
        self.members = members
        self.normalizer = normalizer

    @staticmethod
    def train(samples, y, cfg: TrainCfg, M=5, base_seed=0, bootstrap=True):
        """Train M members. Bootstrap resampling per member sharpens epistemic uncertainty in
        sparse / off-manifold regions (members disagree where data is thin) — the signal both
        V2.4's OOD test and V2.1's calibration depend on. The normalizer is fit once on the full
        training descriptors so all members share a feature scaling."""
        y = np.asarray(y, float)
        _, _, X_desc_full = featurize(samples)
        normalizer = Normalizer.fit(X_desc_full)
        members = []
        for m in range(M):
            if bootstrap:
                rng = np.random.default_rng(base_seed + 100 * m)
                idx = rng.integers(0, len(samples), size=len(samples))
                s_m, y_m = [samples[i] for i in idx], y[idx]
            else:
                s_m, y_m = samples, y
            model, _ = train_member(s_m, y_m, cfg, seed=base_seed + 100 * m,
                                    normalizer=normalizer)
            members.append(model)
        return Ensemble(members, normalizer)

    @torch.no_grad()
    def predict(self, samples):
        """Returns dict: mean, u (total std), epistemic, aleatoric (all numpy, len B)."""
        X_nodes, E, X_desc = featurize(samples)
        Xd = self.normalizer(X_desc)
        mus, vars = [], []
        for model in self.members:
            model.eval()
            mu, logvar = model(X_nodes, E, Xd)
            mus.append(mu); vars.append(torch.exp(logvar.clamp(-8.0, 4.0)))
        mus = torch.stack(mus); vars = torch.stack(vars)               # (M,B)
        mean = mus.mean(0)
        epistemic = mus.var(0, unbiased=False)
        aleatoric = vars.mean(0)
        u = torch.sqrt(epistemic + aleatoric)
        return {k: v.cpu().numpy() for k, v in dict(
            mean=mean, u=u, epistemic=torch.sqrt(epistemic), aleatoric=torch.sqrt(aleatoric)).items()}

    def state(self):
        return [m.state_dict() for m in self.members]

    def load(self, states):
        for m, s in zip(self.members, states):
            m.load_state_dict(s)
        return self


class EnvelopeDetector:
    """The learned tier's VALIDITY ENVELOPE in descriptor space (ARCHITECTURE §III.5).

    "Envelope = on-distribution region of the learned tier; envelope-exit -> fallback." Fit a
    Gaussian on the training descriptors; score a cell by Mahalanobis distance. OOD cells (off-
    manifold contrast/topology) score far outside the envelope. This is the descriptor-side OOD
    trigger; the ensemble `u` is the corroborating prediction-side signal.
    """

    def __init__(self, mu, sd):
        self.mu = mu
        self.sd = sd

    @staticmethod
    def fit(train_samples, floor=1e-6):
        # Per-channel z-score (robust to the descriptor's collinear isotropic channels, where a
        # full-covariance Mahalanobis goes singular). Score = max standardized deviation.
        D = np.stack([vc.descriptor(s.grid, s.materials) for s in train_samples])
        sd = D.std(0)
        sd = np.where(sd < floor, np.inf, sd)        # constant channels carry no OOD signal
        return EnvelopeDetector(D.mean(0), sd)

    def score(self, samples):
        D = np.stack([vc.descriptor(s.grid, s.materials) for s in samples])
        return np.abs((D - self.mu) / self.sd).max(axis=1)


def fallback_flags(samples, env, z_thresh):
    """The learned tier's multi-signal OOD/fallback trigger (V2.4).

    Mirrors the operator schema's "fallback on envelope-exit OR residual-spike" plus V2.2's
    connectivity guard: a cell falls back to the analytic/RVE tier if its descriptor leaves the
    validity envelope (max-z > z_thresh) OR a percolating soft seam is detected (the connectivity
    blind spot the volume-fraction descriptor cannot see). Returns (flags, env_score) bools.
    """
    env_score = env.score(samples)
    # cells are y-extruded by construction, so the meaningful crack-percolation is in the load
    # plane (x,z); the y-axis spans trivially and is excluded.
    perc = np.array([vc.percolates(s.grid, s.materials, axis=0)
                     or vc.percolates(s.grid, s.materials, axis=2) for s in samples])
    return (env_score > z_thresh) | perc, env_score


if __name__ == "__main__":
    # DNS-free smoke test: fit a smooth synthetic retention field over random cells, confirm
    # (a) the ensemble fits in-family, (b) uncertainty rises on shifted (OOD-like) inputs.
    np.set_printoptions(precision=3, suppress=True)
    print(f"device: {DEVICE}")
    rng = np.random.default_rng(0)
    n = 16

    train = vc.family_battery(n, rng, 40)
    test = vc.family_battery(n, rng, 20)
    ood = vc.ood_battery(n, rng, 6)

    # synthetic ground truth: smooth decreasing function of (depth, log-contrast) — stands in
    # for the DNS retention so __main__ stays fast (no FE solves).
    def synth(s):
        depth, contrast = s.theta if s.kind == "char_wedge" else (0.9, s.theta[-1])
        return float(np.clip(0.9 - 0.5 * depth - 0.12 * np.log10(contrast), 0.02, 1.0))

    y_tr = [synth(s) for s in train]
    cfg = TrainCfg(epochs=250, physics=True)
    ens = Ensemble.train(train, y_tr, cfg, M=5, base_seed=0)

    pred_tr = ens.predict(train)
    pred_te = ens.predict(test)
    pred_ood = ens.predict(ood)
    y_te = np.array([synth(s) for s in test])
    rel_te = np.abs(pred_te["mean"] - y_te) / np.abs(y_te)
    print(f"1) in-family fit:  median rel err = {np.median(rel_te):.3f}")
    print(f"2) uncertainty:    in-family median u = {np.median(pred_te['u']):.4f}; "
          f"OOD median u = {np.median(pred_ood['u']):.4f}")
    assert np.median(rel_te) < 0.15, "ensemble should fit the smooth in-family target"
    assert np.median(pred_ood["u"]) > np.median(pred_te["u"]), "u must rise OOD"

    # 2b) multi-signal fallback trigger (envelope-exit OR percolation): the seam is a
    #     connectivity OOD the volume-fraction envelope is blind to (V2.2), so the combined
    #     trigger is needed to catch all OOD categories.
    env = EnvelopeDetector.fit(train)
    thr = np.quantile(env.score(test), 0.95)
    flags_te, _ = fallback_flags(test, env, thr)
    flags_ood, _ = fallback_flags(ood, env, thr)
    print(f"2b) fallback trigger: OOD detected {flags_ood.mean()*100:.0f}%, "
          f"in-family false-positive {flags_te.mean()*100:.0f}% (z_thr={thr:.2f})")
    assert flags_ood.mean() >= 0.99, "combined trigger must flag essentially all OOD cells"
    assert flags_te.mean() <= 0.10, "in-family false-positive rate must stay low"

    # 3) save/load reproducibility
    states = ens.state()
    ens2 = Ensemble([GraphPIGN().to(DEVICE) for _ in range(5)], ens.normalizer).load(states)
    p2 = ens2.predict(test)
    assert np.allclose(p2["mean"], pred_te["mean"], atol=1e-10), "reload must reproduce"
    print("3) save/load reproduces predictions: OK")
    print("\nALL surrogate_gnn self-checks PASSED")
