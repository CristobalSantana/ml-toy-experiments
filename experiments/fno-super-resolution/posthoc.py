"""
posthoc.py -- Two checks added after the results were in. Not
pre-registered, and reported as such.

    python posthoc.py

  1. Where is the zero-shot FNO right? Its error on the points of the
     512-point grid that are NOT on the 64-point training grid - points it
     was never supervised on - against interpolation and against the truth
     itself sampled at 64 and interpolated. Seed 0, the 16 held-out starts
     saved in examples.npz.
  2. Where do the training set's fronts sit inside a 64-point cell? If the
     steepest point of each training solution falls at every fraction of a
     cell, then across the training set the coarse grid samples a front at
     every offset - which is how a translation-equivariant operator could
     learn the profile of a front that no single 64-point sample shows.
     Data only, no model.

Writes outputs/posthoc.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE.parent.parent / "generators" / "burgers_1d" / "outputs" / "burgers.npz"


def interp_linear(u, n):
    m = u.shape[-1]
    pos = np.arange(n) * m / n
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    return u[..., i0 % m] * (1 - w) + u[..., (i0 + 1) % m] * w


def main() -> None:
    z = np.load(OUT / "examples.npz")
    d = np.load(DATA)
    nus = [float(v) for v in d["viscosities"]]
    new = np.arange(512) % 8 != 0                  # not on the 64-point grid
    out = {"unsupervised_points": {}, "front_offsets": {}}

    print("1. relative L2 error on the 448 of 512 points that are not on the 64-point grid")
    print("   (seed 0, 16 held-out starts)\n")
    print(f"   {'nu':<6}{'zero-shot':>11}{'interp. of':>12}{'trained':>10}{'truth at 64,':>15}")
    print(f"   {'':<6}{'':>11}{'its 64 pts':>12}{'at 256':>10}{'interpolated':>15}")
    for nu in nus:
        t = z[f"truth_nu{nu:g}"]

        def rel(p):
            diff = (p - t)[:, new]
            return float(np.mean(np.linalg.norm(diff, axis=1) / np.linalg.norm(t[:, new], axis=1)))

        row = {"zero_shot": rel(z[f"nu{nu:g}_fno64_zero_shot_512"]),
               "interp_own_64": rel(interp_linear(z[f"nu{nu:g}_fno64_zero_shot_64"], 512)),
               "trained_at_256": rel(z[f"nu{nu:g}_fno256_zero_shot_512"]),
               "truth_64_interp": rel(interp_linear(t[:, ::8], 512))}
        out["unsupervised_points"][f"{nu:g}"] = row
        print(f"   {nu:<6g}{row['zero_shot']:>11.4f}{row['interp_own_64']:>12.4f}"
              f"{row['trained_at_256']:>10.4f}{row['truth_64_interp']:>15.4f}")

    print("\n2. where each training solution's steepest point sits inside its 64-point cell")
    for nu in nus:
        uT = d[f"uT_nu{nu:g}"][:1000].astype(float)
        step = np.diff(uT, axis=1, append=uT[:, :1])
        i = np.argmin(step, axis=1)                     # steepest descent, 512-point grid
        frac = ((i + 0.5) % 8) / 8                      # 0..1 across the coarse cell
        counts, _ = np.histogram(frac, bins=8, range=(0, 1))
        out["front_offsets"][f"{nu:g}"] = counts.tolist()
        print(f"   nu {nu:<5g} per eighth of a cell: {counts.tolist()}   "
              f"(uniform: {len(frac) / 8:.0f} each)")

    (OUT / "posthoc.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'posthoc.json'}")


if __name__ == "__main__":
    main()
