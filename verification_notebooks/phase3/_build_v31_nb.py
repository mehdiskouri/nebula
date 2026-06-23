"""Builds V3_1_buoyant_plume.ipynb via nbformat (Tier 3, the flame spine).

Thresholds FROZEN here (repo practice: set below measured margins; protocol §1). V3.1's claim
is deliberately scoped to what a correct, conservative, deterministic Boussinesq transport MUST
do — be divergence-free, conserve transported buoyancy exactly, rise with the right sign, and
carry buoyancy upward as a coherent plume that widens by entrainment — verified against the
independent MTT integral-plume oracle. The turbulent self-similar EXPONENTS (w∝z^-1/3, ΔT∝z^-5/3)
are REPORTED, not gated: a laminar coarse solver has no turbulent entrainment, so reproducing the
turbulent constants is out of scope (a documented standing constraint, like V1.2/V1.3's CONSTRAIN).

Run: .venv/bin/python verification_notebooks/phase3/_build_v31_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []

C.append(new_markdown_cell(
    "# V3.1 — Buoyant plume: the flame solver's transport core  **TIER 3 / flame spine**\n\n"
    "**Claim (pre-registered).** The Boussinesq projection–advection solver "
    "(`nebula.operators.flow`) is a correct incompressible buoyant transport: it is "
    "divergence-free after projection, conserves transported buoyancy **exactly** (advection = "
    "conserved-flux staging on the bus), rises with the correct buoyancy sign, and carries its "
    "source buoyancy upward as a coherent plume that **widens by entrainment** — the physics "
    "Phase-0's in-place 0-D combustion lacked, and the prerequisite for a flame that stands OFF "
    "the fuel (V3.2) and smoke that rises (V3.3).\n\n"
    "**Why load-bearing.** Blocker #1: Phase-0 had *no flame* — the volatile gas never moved, so "
    "combustion burned inside the wood voxels and the 'fire' was orange spheres stuck to the bark. "
    "A buoyant velocity field that transports heat/volatiles/soot is the missing mechanism. It must "
    "obey the two guardrails: buoyancy enters as a **potential** (Boussinesq body force) made "
    "divergence-free by **pressure projection** (energy-stable Helmholtz–Hodge), never a raw force; "
    "and transport is **conservative-flux** so the V0.3 conservation audit extends to it.\n\n"
    "**Independent oracle.** The Morton–Taylor–Turner (MTT) integral plume model "
    "(`plume_analytic.py`) — a 1-D ODE entrainment model with no shared code or discretization with "
    "the 3-D PDE solver. Its solver-agnostic predictions: the buoyancy flux is **conserved with "
    "height** (dF/dz=0) and the plume **widens** (b∝z); its turbulent centerline exponents "
    "(w∝z^-1/3, ΔT∝z^-5/3) are the self-similar target reported for comparison.\n\n"
    "**Pre-registered pass criteria (frozen below measured margins):**\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| C1 | incompressibility: post-projection max\\|∇·u\\|·dx (random divergent field) | < 1e-8 |\n"
    "| C2 | transport conservation: passive scalar Σ over 40 steps | rel < 1e-10 |\n"
    "| C3 | buoyancy sign: hot-patch heat center-of-mass rise / cold-patch sink | hot ≥ +2 cells AND cold ≤ −1 cell |\n"
    "| C4 | **exact buoyancy budget**: ∫(T−T_ref)dV vs Σ tracked source input | rel < 1e-8 |\n"
    "| C5 | coherent rising plume: buoyancy flux F(z)>0 everywhere & spread; width growth | F>0 all AND spread < 0.65 AND b_top/b_bot ≥ 2 |\n"
    "| C6 | determinism: re-run bitwise-identical (fixed-order CPU reductions, V0.5) | identical |\n\n"
    "**Reported (not gated) — the honest laminar gap.** Centerline w(z), ΔT(z) log-log slopes vs "
    "the MTT turbulent targets −1/3, −5/3. A laminar coarse solver under-entrains, so it will not "
    "reproduce the turbulent constants; this is a documented Tier-3 standing constraint, and the "
    "load-bearing transport facts (C1–C6) hold regardless."
))

C.append(new_code_cell(
    "import sys, pathlib\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "ROOT = pathlib.Path.cwd().resolve()\n"
    "while ROOT.name and not (ROOT / 'src').exists():\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT / 'src' / 'implementation'))\n"
    "sys.path.insert(0, str(ROOT / 'src' / 'verification' / 'oracles'))\n"
    "from nebula.operators import flow\n"
    "import plume_analytic as pa\n"
    "np.seterr(all='ignore')\n"
    "results = {}\n"
    "print('flow + MTT oracle loaded')"
))

C.append(new_markdown_cell("## C1 — incompressibility (projection is exact)"))
C.append(new_code_cell(
    "p = flow.FlowParams(dx=1.0, beta=1.0/300.0, g=9.81, T_ref=300.0)\n"
    "shape = (24, 24, 24)\n"
    "rng = np.random.default_rng(0)\n"
    "u, v, w = flow.zero_velocity(shape)\n"
    "u += rng.standard_normal(u.shape); v += rng.standard_normal(v.shape); w += rng.standard_normal(w.shape)\n"
    "u[0]=u[-1]=0; v[:,0]=v[:,-1]=0; w[:,:,0]=w[:,:,-1]=0\n"
    "u, v, w, d0, d1 = flow.project(u, v, w, p.dx)\n"
    "results['C1_div'] = d1\n"
    "print(f'max|div|: {d0:.3e} -> {d1:.3e}   (C1 < 1e-8: {d1 < 1e-8})')"
))

C.append(new_markdown_cell("## C2 — transport conservation (advection = conserved-flux staging)"))
C.append(new_code_cell(
    "s = np.zeros(shape); s[8:16, 8:16, 8:16] = 1.0\n"
    "u, v, w = flow.zero_velocity(shape)\n"
    "u += 0.5; v += 0.3\n"
    "u[0]=u[-1]=0; v[:,0]=v[:,-1]=0; w[:,:,0]=w[:,:,-1]=0\n"
    "u, v, w, *_ = flow.project(u, v, w, p.dx)\n"
    "tot0 = s.sum()\n"
    "for _ in range(40):\n"
    "    s = flow.advect_scalar(s, u, v, w, 0.4, p.dx)\n"
    "results['C2_cons'] = abs(s.sum()-tot0)/tot0\n"
    "print(f'Σs {tot0:.6f} -> {s.sum():.6f}   rel {results[\"C2_cons\"]:.2e}  (C2 < 1e-10: {results[\"C2_cons\"]<1e-10})')"
))

C.append(new_markdown_cell("## C3 — buoyancy sign (hot rises, cold sinks)"))
C.append(new_code_cell(
    "def heat_com(field, sign, p):\n"
    "    d = np.clip(sign*(field - p.T_ref), 0, None); z=(np.arange(field.shape[2])+0.5)\n"
    "    return float((d.sum(axis=(0,1))*z).sum()/(d.sum()+1e-30))\n"
    "# hot patch at base -> rises\n"
    "sh=(24,24,40); u,v,w=flow.zero_velocity(sh); sc={'T':np.full(sh,300.0)}\n"
    "ix,iy=np.meshgrid(np.arange(24),np.arange(24),indexing='ij'); disk=((ix-12)**2+(iy-12)**2)<=4\n"
    "hot0=None\n"
    "for n in range(50):\n"
    "    sc['T'][disk,0:2]=800.0\n"
    "    u,v,w,sc,info=flow.step(u,v,w,sc,p,0.5)\n"
    "    if n==5: hot0=heat_com(sc['T'],+1,p)\n"
    "hot1=heat_com(sc['T'],+1,p)\n"
    "# cold patch at top -> sinks\n"
    "u,v,w=flow.zero_velocity(sh); sc={'T':np.full(sh,300.0)}; cold0=None\n"
    "for n in range(50):\n"
    "    sc['T'][disk,-2:]=150.0\n"
    "    u,v,w,sc,info=flow.step(u,v,w,sc,p,0.5)\n"
    "    if n==5: cold0=heat_com(sc['T'],-1,p)\n"
    "cold1=heat_com(sc['T'],-1,p)\n"
    "results['C3_hot_rise']=hot1-hot0; results['C3_cold_sink']=cold1-cold0\n"
    "print(f'hot CoM {hot0:.2f}->{hot1:.2f} (Δ{hot1-hot0:+.2f}); cold CoM {cold0:.2f}->{cold1:.2f} (Δ{cold1-cold0:+.2f})')\n"
    "print(f'C3: hot rise≥+2 {hot1-hot0>=2}, cold sink≤-1 {cold1-cold0<=-1}')"
))

C.append(new_markdown_cell(
    "## C4/C5 — the plume: exact buoyancy budget, conserved upward flux, entrainment widening\n"
    "Steady plume from a heated base disk; track the **exact** source-buoyancy input (the only "
    "thing that changes ∫T besides conservative advection), then time-average the height-resolved "
    "buoyancy flux F(z)=∫w·ΔT dA and the RMS plume width b(z)."
))
C.append(new_code_cell(
    "def plume_width(T, p):\n"
    "    nx,ny,nz=T.shape\n"
    "    ix,iy=np.meshgrid((np.arange(nx)-nx/2+0.5)*p.dx,(np.arange(ny)-ny/2+0.5)*p.dx,indexing='ij')\n"
    "    r2=(ix*ix+iy*iy)[:,:,None]; dT=np.clip(T-p.T_ref,0,None)\n"
    "    return np.sqrt((r2*dT).sum(axis=(0,1))/(dT.sum(axis=(0,1))+1e-30))\n"
    "nx=32; nz=80; shape=(nx,nx,nz)\n"
    "u,v,w=flow.zero_velocity(shape); sc={'T':np.full(shape,300.0)}\n"
    "cx=nx//2; ix,iy=np.meshgrid(np.arange(nx),np.arange(nx),indexing='ij'); disk=((ix-cx)**2+(iy-cx)**2)<=4\n"
    "dV=p.dx**3; source_input=0.0; Fh=[]; bh=[]; ph=[]; maxdiv=0.0\n"
    "for n in range(200):\n"
    "    before=sc['T'][disk,0:2].copy()\n"
    "    sc['T'][disk,0]=900.0; sc['T'][disk,1]=600.0\n"
    "    source_input += float((sc['T'][disk,0:2]-before).sum())*dV   # exact tracked deposit\n"
    "    u,v,w,sc,info=flow.step(u,v,w,sc,p,0.5); maxdiv=max(maxdiv,info['div_after'])\n"
    "    if n>=160:\n"
    "        pr=flow.centerline_profiles(sc,u,v,w,p); Fh.append(pr['F']); bh.append(plume_width(sc['T'],p)); ph.append(pr['dT'])\n"
    "anomaly = float(np.clip(sc['T']-p.T_ref,None,None).sum())*dV\n"
    "results['C4_budget'] = abs(anomaly-source_input)/abs(source_input)\n"
    "z=flow.centerline_profiles(sc,u,v,w,p)['z']; F=np.mean(Fh,0); b=np.mean(bh,0); dT=np.mean(ph,0)\n"
    "zlo,zhi=10,50; m=(z>=zlo)&(z<=zhi)\n"
    "results['C5_Fpos']=bool((F[m]>0).all()); results['C5_Fspread']=float((F[m].max()-F[m].min())/abs(F[m].mean()))\n"
    "results['C5_widen']=float(b[m][-1]/(b[m][0]+1e-9)); results['C5_maxdiv']=maxdiv\n"
    "print(f'C4 buoyancy budget: ∫anomaly {anomaly:.1f} vs source {source_input:.1f}  rel {results[\"C4_budget\"]:.2e}')\n"
    "print(f'C5 flux F>0 all: {results[\"C5_Fpos\"]}  spread {results[\"C5_Fspread\"]:.3f}  widen b_top/b_bot {results[\"C5_widen\"]:.2f}  max|div| {maxdiv:.1e}')"
))

C.append(new_markdown_cell("## Reported (not gated) — laminar centerline slopes vs MTT turbulent targets"))
C.append(new_code_cell(
    "sl_dT = pa.loglog_slope(z, np.clip(dT,1e-6,None), zlo, zhi)\n"
    "sl_b  = pa.loglog_slope(z, np.clip(b,1e-6,None), zlo, zhi)\n"
    "results['rep_slope_dT']=sl_dT; results['rep_slope_b']=sl_b\n"
    "print(f'centerline ΔT slope {sl_dT:+.3f}  (MTT turbulent target -5/3={-5/3:+.3f})')\n"
    "print(f'plume width   slope {sl_b:+.3f}  (MTT target +1)')\n"
    "print('NOTE: laminar coarse solver under-entrains; turbulent exponents not reproduced (documented).')"
))

C.append(new_markdown_cell("## C6 — determinism (the program IS the asset)"))
C.append(new_code_cell(
    "def run():\n"
    "    sh=(20,20,40); uu,vv,ww=flow.zero_velocity(sh); s2={'T':np.full(sh,300.0)}\n"
    "    ix,iy=np.meshgrid(np.arange(20),np.arange(20),indexing='ij'); dk=((ix-10)**2+(iy-10)**2)<=4\n"
    "    for _ in range(10):\n"
    "        s2['T'][dk,0:2]=850.0\n"
    "        uu,vv,ww,s2,_=flow.step(uu,vv,ww,s2,p,0.5)\n"
    "    return s2['T']\n"
    "a,b2=run(),run()\n"
    "results['C6_bitexact']=bool(np.array_equal(a.view(np.uint64), b2.view(np.uint64)))\n"
    "print(f'C6 re-run bitwise-identical: {results[\"C6_bitexact\"]}')"
))

C.append(new_markdown_cell("## Figure + verdict"))
C.append(new_code_cell(
    "fig, ax = plt.subplots(2, 3, figsize=(15, 8))\n"
    "# (0,0) plume ΔT mid-slice\n"
    "sl = (sc['T'][:, nx//2, :].T - p.T_ref)\n"
    "im=ax[0,0].imshow(np.clip(sl,0,None), origin='lower', aspect='auto', cmap='inferno'); ax[0,0].set_title('plume ΔT (x–z mid-slice)'); fig.colorbar(im,ax=ax[0,0])\n"
    "# (0,1) centerline ΔT vs z with MTT shape\n"
    "ax[0,1].plot(z, dT, 'o-', ms=3, label='solver ΔT(z)')\n"
    "zz=z[(z>=zlo)&(z<=zhi)]; ax[0,1].plot(zz, dT[(z>=zlo)&(z<=zhi)][0]*(zz/zz[0])**(-5/3), '--', label='MTT z^-5/3 (turbulent)')\n"
    "ax[0,1].set_title('centerline ΔT vs height'); ax[0,1].set_xlabel('z'); ax[0,1].legend()\n"
    "# (0,2) buoyancy flux F(z) conservation\n"
    "ax[0,2].plot(z, F, 'o-', ms=3); ax[0,2].axvspan(zlo,zhi,color='g',alpha=0.1); ax[0,2].set_title(f'buoyancy flux F(z) (spread {results[\"C5_Fspread\"]:.2f}, MTT dF/dz=0)'); ax[0,2].set_xlabel('z')\n"
    "# (1,0) width b(z)\n"
    "ax[1,0].plot(z, b, 'o-', ms=3); ax[1,0].set_title(f'plume width b(z) (widen ×{results[\"C5_widen\"]:.1f}, MTT b∝z)'); ax[1,0].set_xlabel('z')\n"
    "# (1,1) criteria bars\n"
    "labs=['C1 div','C2 cons','C4 budget']; vals=[results['C1_div'],results['C2_cons'],results['C4_budget']]\n"
    "ax[1,1].bar(labs, np.log10(np.array(vals)+1e-30)); ax[1,1].axhline(-8,color='r',ls='--'); ax[1,1].set_title('log10 residuals (< -8 / -10 pass)')\n"
    "# (1,2) verdict text\n"
    "C1=results['C1_div']<1e-8; C2=results['C2_cons']<1e-10\n"
    "C3=(results['C3_hot_rise']>=2) and (results['C3_cold_sink']<=-1)\n"
    "C4=results['C4_budget']<1e-8\n"
    "C5=results['C5_Fpos'] and results['C5_Fspread']<0.65 and results['C5_widen']>=2.0\n"
    "C6=results['C6_bitexact']\n"
    "allpass=C1 and C2 and C3 and C4 and C5 and C6\n"
    "txt='\\n'.join([f'C1 incompressible: {C1}',f'C2 conservation: {C2}',f'C3 buoyancy sign: {C3}',f'C4 buoyancy budget: {C4}',f'C5 coherent plume: {C5}',f'C6 determinism: {C6}','',f'V3.1 VERDICT: {\"PASS\" if allpass else \"FAIL\"}'])\n"
    "ax[1,2].axis('off'); ax[1,2].text(0.02,0.98,txt,va='top',ha='left',fontsize=12,family='monospace')\n"
    "fig.suptitle('V3.1 — Buoyant plume (flame solver transport core)', fontsize=14)\n"
    "fig.tight_layout(); fig.savefig('results/V3_1_buoyant_plume.png', dpi=110, bbox_inches='tight')\n"
    "print('saved results/V3_1_buoyant_plume.png')\n"
    "assert C1 and C2 and C3 and C4 and C5 and C6, 'V3.1 criteria not all met'\n"
    "print('\\nV3.1 PASS — buoyant transport core verified against the MTT oracle.')"
))

nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "V3_1_buoyant_plume.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}")
