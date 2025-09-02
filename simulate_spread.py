import pandas as pd
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
"""
Run a simple risk-guided spread simulation and export an animated GIF.
Uses the latest risk field to evolve a probabilistic cellular automaton.
"""

grid = pd.read_csv("data/grid_2km.csv")
risk = pd.read_csv("data/risk_scores.csv", parse_dates=["date"])
xs = np.sort(grid["xmin"].unique())
ys = np.sort(grid["ymin"].unique())
nx = len(xs)
ny = len(ys)
d0 = risk["date"].max()
dfd = risk[risk["date"]==d0].merge(grid[["cell_id","xmin","ymin"]], on="cell_id")
R = np.zeros((ny,nx))
lx = {x:i for i,x in enumerate(xs)}
ly = {y:j for j,y in enumerate(ys)}
for _, r in dfd.iterrows():
    i = lx[r["xmin"]]
    j = ly[r["ymin"]]
    R[j,i] = float(r["risk"])
k = max(5, int(0.01 * R.size))
seed = np.zeros_like(R)
seed.ravel()[np.argsort(R.ravel())[::-1][:k]] = 1.0
burn = seed.copy()
wind_dx, wind_dy = 1.0, 0.0
frames = []
Path("figures").mkdir(parents=True, exist_ok=True)
for t in range(24):
    nb = np.zeros_like(burn)
    nb[:-1,:] += burn[1:,:]
    nb[1:,:] += burn[:-1,:]
    nb[:,:-1] += burn[:,1:]
    nb[:,1:] += burn[:,:-1]
    adv = np.zeros_like(burn)
    if wind_dx > 0:
        adv[:,:-1] += burn[:,1:]
    if wind_dx < 0:
        adv[:,1:] += burn[:,:-1]
    if wind_dy > 0:
        adv[:-1,:] += burn[1:,:]
    if wind_dy < 0:
        adv[1:,:] += burn[:-1,:]
    p = 0.15 * nb + 0.10 * adv + 0.75 * R
    p = np.clip(p, 0, 1)
    new_burn = (np.random.rand(*burn.shape) < p).astype(float)
    burn = np.clip(burn + new_burn, 0, 1)
    plt.figure(figsize=(6,5))
    plt.imshow(burn, origin="lower", aspect="equal")
    plt.title(f"Risk-guided spread simulation t={t}")
    plt.axis("off")
    frame_path = Path("figures") / f"sim_{t:02d}.png"
    plt.tight_layout()
    plt.savefig(frame_path, dpi=140)
    plt.close()
    frames.append(imageio.imread(frame_path.as_posix()))
imageio.mimsave("figures/risk_sim.gif", frames, duration=0.2)