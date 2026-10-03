"""
generate.py -- Viscous Burgers' equation on a periodic line: random smooth
starts, solved to a fixed time, at three viscosities.

Physics
-------
    u_t + u u_x = nu u_xx          x in [0, 2 pi), periodic

The nonlinear term steepens every downhill slope into a front; viscosity
smears the front back out to a width of about 4 nu / (jump in u). With
nu = 0.1 the fronts are broad and the solution at t = 1 is smooth on any
reasonable grid. With nu = 0.01 they are ten times narrower, and most of
the solution's detail sits at wavenumbers a coarse grid cannot represent.
That is the point of the three viscosities: the same equation, the same
starts, and a solution whose fine-scale content is turned up by a dial.

The starts are band-limited: Fourier modes 1 to 16 only, amplitude falling
as k^-2, zero mean, normalised to RMS 1. A grid of 64 points samples every
one of them exactly. Whatever fine structure the solution has at t = 1 was
made by the equation, not put there by the initial condition - so a model
that sees the start on 64 points has, in principle, all the information
there is.

Method
------
Pseudo-spectral in space on 2048 points with the 2/3 dealiasing rule,
fourth-order exponential time differencing in time (ETDRK4, Kassam &
Trefethen 2005): the stiff diffusion is integrated exactly, the nonlinear
term to fourth order. Stored on every fourth grid point, 512, which nests
exactly the 256-, 128- and 64-point grids the experiments use.

Three checks against known results run on every generation:
  1. the exact solution. The Cole-Hopf transform turns Burgers into the heat
     equation, which gives u(x, t) as a ratio of two integrals (the Hopf
     formula). Evaluated by quadrature in log-sum-exp form, so it stays
     stable at small viscosity, and compared with the solver point by point
  2. mass. The mean of u is conserved exactly by the equation
  3. the energy law. d/dt (1/2) int u^2 = - nu int u_x^2, step by step

Deterministic given the seed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs"


@dataclass(frozen=True)
class BurgersParams:
    n_samples: int = 1200            # initial conditions, shared by every viscosity
    viscosities: tuple = (0.1, 0.03, 0.01)
    t_end: float = 1.0
    n_fine: int = 2048               # solver grid
    n_store: int = 512               # stored grid: every (n_fine / n_store)-th point
    k_max: int = 16                  # the start uses Fourier modes 1..k_max only
    spectrum_decay: float = 2.0      # amplitude of mode k ~ k^-decay
    dt: float = 5e-4                 # ETDRK4 step
    n_exact_check: int = 24          # samples checked against the Hopf formula
    n_energy_check: int = 32         # samples whose energy is tracked every step
    seed: int = 20261003

    def as_dict(self) -> dict:
        d = asdict(self)
        d["viscosities"] = list(self.viscosities)
        return d


# --------------------------------------------------------------------------
# initial conditions
# --------------------------------------------------------------------------

def initial_conditions(p: BurgersParams) -> np.ndarray:
    """(n_samples, n_fine): random band-limited fields, zero mean, RMS 1."""
    rng = np.random.default_rng(p.seed)
    k = np.arange(1, p.k_max + 1)
    coef = np.zeros((p.n_samples, p.n_fine // 2 + 1), dtype=complex)
    coef[:, 1:p.k_max + 1] = ((rng.standard_normal((p.n_samples, p.k_max))
                               + 1j * rng.standard_normal((p.n_samples, p.k_max)))
                              * k ** -p.spectrum_decay)
    u0 = np.fft.irfft(coef, n=p.n_fine, axis=1)
    return u0 / np.sqrt(np.mean(u0 ** 2, axis=1, keepdims=True))


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------

def etdrk4_coefficients(lin: np.ndarray, dt: float, m: int = 32):
    """Kassam & Trefethen's contour-integral evaluation of the ETDRK4
    coefficients, stable where dt * lin is near zero."""
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * lin[:, None] + r[None, :]
    q = dt * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
    f1 = dt * np.real(np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr ** 2)) / lr ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * lr - lr ** 2 + np.exp(lr) * (4 - lr)) / lr ** 3, axis=1))
    return np.exp(dt * lin), np.exp(dt * lin / 2), q, f1, f2, f3


def solve(u0: np.ndarray, nu: float, p: BurgersParams, track: int = 0):
    """Integrate every row of u0 to t_end. Returns u(t_end) on the fine grid,
    and for the first `track` rows the energy and dissipation at every step."""
    n = p.n_fine
    k = np.arange(n // 2 + 1, dtype=float)
    lin = -nu * k ** 2
    dealias = k < n / 3                                  # the 2/3 rule
    g = -0.5j * k * dealias
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(lin, p.dt)

    def nonlinear(v):
        u = np.fft.irfft(v * dealias, n=n, axis=1)
        return g * np.fft.rfft(u * u, axis=1)

    def energy(v):
        u = np.fft.irfft(v[:track], n=n, axis=1)
        ux = np.fft.irfft(1j * k * v[:track], n=n, axis=1)
        return np.pi * np.mean(u ** 2, axis=1), 2 * np.pi * nu * np.mean(ux ** 2, axis=1)

    v = np.fft.rfft(u0, axis=1)
    n_steps = int(round(p.t_end / p.dt))
    E_hist, D_hist = [], []
    for step in range(n_steps + 1):
        if track:
            e, d = energy(v)
            E_hist.append(e); D_hist.append(d)
        if step == n_steps:
            break
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    return np.fft.irfft(v, n=n, axis=1), np.array(E_hist), np.array(D_hist)


# --------------------------------------------------------------------------
# checks against known results
# --------------------------------------------------------------------------

def hopf_solution(u0: np.ndarray, nu: float, t: float, x_eval: np.ndarray) -> np.ndarray:
    """Exact u(x, t) from the Hopf formula, for one start sampled on [0, 2 pi).

        u(x, t) = int ((x - y) / t) w(y) dy / int w(y) dy
        w(y)    = exp(-[(x - y)^2 / (2 t) + U0(y)] / (2 nu))

    with U0 the antiderivative of u0, periodic because u0 has zero mean. The
    integral runs over three periods, enough for every characteristic that
    can reach x by time t, and the exponent is shifted by its maximum before
    exponentiating - at nu = 0.01 it spans hundreds of units.
    """
    n = len(u0)
    k = np.arange(n // 2 + 1)
    uh = np.fft.rfft(u0)
    Uh = np.zeros_like(uh)
    Uh[1:] = uh[1:] / (1j * k[1:])
    U0 = np.fft.irfft(Uh, n=n)
    y = np.arange(n) * 2 * np.pi / n
    y3 = np.concatenate([y - 2 * np.pi, y, y + 2 * np.pi])
    U3 = np.tile(U0, 3)
    out = np.empty(len(x_eval))
    for i, x in enumerate(x_eval):
        expo = -((x - y3) ** 2 / (2 * t) + U3) / (2 * nu)
        w = np.exp(expo - expo.max())
        out[i] = np.sum((x - y3) / t * w) / np.sum(w)
    return out


def spectral_tail(u: np.ndarray, cutoff: int) -> float:
    """Share of the field's energy at wavenumbers above `cutoff`, averaged
    over samples. With cutoff = 32 it is what a 64-point grid cannot hold."""
    e = np.abs(np.fft.rfft(u, axis=1)) ** 2
    e[:, 1:] *= 2
    return float(np.mean(e[:, cutoff + 1:].sum(axis=1) / e.sum(axis=1)))


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def generate(p: BurgersParams) -> dict:
    u0 = initial_conditions(p)
    stride = p.n_fine // p.n_store
    x_store = np.arange(p.n_store) * 2 * np.pi / p.n_store
    out = {"u0": u0[:, ::stride], "x": x_store, "runs": {}}
    for nu in p.viscosities:
        started = time.time()
        uT, E, D = solve(u0, nu, p, track=p.n_energy_check)
        # 1. the exact solution, on the stored grid
        exact_err, exact_scale = [], []
        for j in range(p.n_exact_check):
            ref = hopf_solution(u0[j], nu, p.t_end, x_store)
            exact_err.append(np.abs(uT[j, ::stride] - ref).max())
            exact_scale.append(np.abs(ref).max())
        # 2. mass
        mass = float(np.abs(uT.mean(axis=1) - u0.mean(axis=1)).max())
        # 3. the energy law, away from the first and last step
        dE = np.gradient(E, p.dt, axis=0)
        law_err = float(np.abs(dE[2:-2] + D[2:-2]).max())
        law_rms = float(np.sqrt(np.mean(D[2:-2] ** 2)))
        checks = {
            "hopf_max_abs_error": float(max(exact_err)),
            "hopf_max_abs_error_relative": float(max(exact_err) / np.median(exact_scale)),
            "mass_drift": mass,
            "energy_law_max_error": law_err,
            "energy_law_rms": law_rms,
            "energy_lost_fraction": float(np.median(1 - E[-1] / E[0])),
        }
        # what each grid cannot hold: a grid of n points holds modes up to n/2
        tails = {f"above_{c}": spectral_tail(uT, c) for c in (32, 64, 128, 256)}
        out["runs"][nu] = {"uT": uT[:, ::stride], "checks": checks, "tail": tails,
                           "seconds": time.time() - started}
    return out


def save(data: dict, p: BurgersParams, outdir: Path) -> None:
    """One file: the starts once, and the solution at each viscosity beside
    them. Float32 throughout; the solver's own error is far below that."""
    outdir.mkdir(parents=True, exist_ok=True)
    arrays = {"x": data["x"].astype(np.float32), "u0": data["u0"].astype(np.float32),
              "viscosities": np.array(p.viscosities)}
    for nu, run in data["runs"].items():
        arrays[f"uT_nu{nu:g}"] = run["uT"].astype(np.float32)
    np.savez_compressed(outdir / "burgers.npz", **arrays)
    meta = {"params": p.as_dict(),
            "runs": {f"{nu:g}": {"checks": r["checks"], "spectral_tail": r["tail"]}
                     for nu, r in data["runs"].items()}}
    (outdir / "burgers_params.json").write_text(json.dumps(meta, indent=2))


def plot(data: dict, p: BurgersParams, path: Path) -> None:
    nus = list(data["runs"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = data["x"]
    j = 0
    ax = axes[0]
    ax.plot(x, data["u0"][j], color="black", lw=1.4, label="start, t = 0")
    for nu, c in zip(nus, ("#7aa6c2", "#2f6f99", "#0b2f4a")):
        ax.plot(x, data["runs"][nu]["uT"][j], color=c, lw=1.4,
                label=f"t = {p.t_end:g}, nu = {nu:g}")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title("one start, three viscosities")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    x64 = x[::p.n_store // 64]
    uT = data["runs"][nus[-1]]["uT"][j]
    ax.plot(x, uT, color="#0b2f4a", lw=1.2, label="512 points")
    ax.plot(x64, uT[::p.n_store // 64], "o", ms=3.5, color="#e69f00", label="64 points")
    ax.set_xlabel("x")
    ax.set_title(f"nu = {nus[-1]:g}: what a 64-point grid sees")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[2]
    for nu, c in zip(nus, ("#7aa6c2", "#2f6f99", "#0b2f4a")):
        e = np.abs(np.fft.rfft(data["runs"][nu]["uT"], axis=1)) ** 2
        ax.semilogy(np.arange(e.shape[1])[1:], e.mean(axis=0)[1:], color=c, lw=1.4,
                    label=f"nu = {nu:g}")
    for m, lab in ((32, "64-pt grid"), (128, "256-pt grid")):
        ax.axvline(m, color="grey", ls=":", lw=1)
        ax.text(m + 3, 1e-14, lab, color="grey", fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(bottom=1e-16)
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("mean energy in mode k")
    ax.set_title(f"spectrum at t = {p.t_end:g}; dotted: grid limits")
    ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Viscous Burgers reference solutions", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    for f in fields(BurgersParams):
        if f.name == "viscosities":
            ap.add_argument("--viscosities", type=float, nargs="+", default=list(f.default))
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(ap.parse_args())
    args["viscosities"] = tuple(args["viscosities"])
    p = BurgersParams(**args)

    data = generate(p)
    save(data, p, OUTPUT_DIR)
    plot(data, p, OUTPUT_DIR / "burgers_overview.png")

    print(f"{p.n_samples} starts (modes 1-{p.k_max}, RMS 1), solved to t = {p.t_end:g} "
          f"on {p.n_fine} points, stored on {p.n_store}")
    for nu, r in data["runs"].items():
        c, t = r["checks"], r["tail"]
        print(f"  nu = {nu:g}   ({r['seconds']:.0f} s)")
        print(f"    exact (Hopf) solution, {p.n_exact_check} starts: max |error| "
              f"{c['hopf_max_abs_error']:.1e}  ({c['hopf_max_abs_error_relative']:.1e} of max |u|)")
        print(f"    mass drift {c['mass_drift']:.1e}   energy law: max |dE/dt + nu int u_x^2| "
              f"{c['energy_law_max_error']:.1e} (law RMS {c['energy_law_rms']:.2e}); "
              f"{100 * c['energy_lost_fraction']:.0f}% of the energy dissipated")
        print("    energy share above the highest mode a grid holds:  "
              + "   ".join(f"{2 * int(k.split('_')[1])}-pt {v:.1e}" for k, v in t.items()))
    print(f"-> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
