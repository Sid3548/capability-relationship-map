"""Reconstruct the cliff sweep parquet + cliff analysis + PNG from cliff_run.log
(the process was killed at step 497/498 before writing the parquet; all
per-step NLL/acc data is in the log)."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
log = (REPO / "cliff_run.log").read_text(encoding="utf-8", errors="ignore")

pat = re.compile(
    r"\[sweep\] step=(\d+)/\d+ control=(\w+) seed=(\S+) frac=([\d.]+) "
    r"nll_target=([\d.]+) acc_target=(\S+)")
rows = []
for m in pat.finditer(log):
    st, ctrl, seed, frac, nll, acc = m.groups()
    rows.append({"step": int(st), "control": ctrl,
                 "seed": -1 if seed == "None" else int(seed),
                 "frac_removed": float(frac), "nll_target": float(nll),
                 "acc_target": None if acc == "None" else float(acc)})
df = pd.DataFrame(rows)
out = REPO / "results" / "cliff_sweep_coding.parquet"
df.to_parquet(out)
print(f"reconstructed {len(df)} rows -> {out}")
print(f"max frac reached: {df.frac_removed.max()*100:.2f}%")
print(f"steps: {df.step.max()}")

BASE = 0.819
THRESH = 2 * BASE
low = df[df.control == "low"].sort_values("frac_removed")
# NLL cliff
over = low[low.nll_target > THRESH]
nll_cliff = over.iloc[0].frac_removed if len(over) else None
print(f"\nbaseline coding NLL = {BASE}, 2x threshold = {THRESH:.3f}")
print(f"NLL-cliff (low envelope, first NLL>2x): "
      + (f"{nll_cliff*100:.1f}%" if nll_cliff is not None else f"NOT reached within {low.frac_removed.max()*100:.1f}%"))
print(f"  max low NLL reached = {low.nll_target.max():.3f} at {low.loc[low.nll_target.idxmax(),'frac_removed']*100:.1f}%")

# pass@1 cliff (gen checkpoints)
gen = low[low.acc_target.notna()].sort_values("frac_removed")
print("\nlow-envelope gen checkpoints (frac% -> pass@1):")
for _, r in gen.iterrows():
    print(f"  {r.frac_removed*100:5.1f}% -> {r.acc_target:.2f}")
BASE_ACC = 0.75
collapse = gen[gen.acc_target < 0.5 * BASE_ACC]  # retention < 0.5
pass_cliff = collapse.iloc[0].frac_removed if len(collapse) else None
print(f"\npass@1-cliff (retention<0.5, i.e. pass@1<{0.5*BASE_ACC}): "
      + (f"{pass_cliff*100:.1f}%" if pass_cliff is not None else "not reached"))

# retention PNG to 50%
fig, ax1 = plt.subplots(figsize=(9, 5.5))
for ctrl, color, lab in [("low", "#1f77b4", "low envelope"), ("random", "#7f7f7f", "random (mean)")]:
    sub = df[df.control == ctrl]
    if ctrl == "random":
        agg = sub.groupby("frac_removed", as_index=False).nll_target.mean()
        ax1.plot(agg.frac_removed * 100, agg.nll_target, color=color, label=lab, lw=1.4)
    else:
        s = sub.sort_values("frac_removed")
        ax1.plot(s.frac_removed * 100, s.nll_target, color=color, label=lab, lw=1.4)
ax1.axhline(THRESH, color="red", ls="--", lw=0.9, label=f"2x baseline ({THRESH:.2f})")
ax1.axhline(BASE, color="green", ls=":", lw=0.8, label=f"baseline ({BASE})")
ax1.set_xlabel("% MLP neurons removed (coding target)")
ax1.set_ylabel("coding teacher-forced NLL")
ax1.set_title("Coding cliff sweep to ~50% — envelope vs random")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
fig.tight_layout()
png = REPO / "results" / "cliff_retention_to50.png"
fig.savefig(png, dpi=150)
print(f"\nsaved {png}")
