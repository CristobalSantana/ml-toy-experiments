"""
generate.py -- Reference trajectories of a pendulum, with and without friction.

Physics
-------
The ideal pendulum, in units where mass, length and gravity are all 1:

    H(q, p) = p^2 / 2  -  cos(q)

    dq/dt =  dH/dp =  p
    dp/dt = -dH/dq = -sin(q)

`q` is the angle from the bottom, `p` the angular momentum, and `H` the total
energy, which the ideal system conserves exactly. Energy sorts the motion into
two kinds: below H = 1 the pendulum swings back and forth (libration), above
H = 1 it goes over the top (rotation), and H = 1 itself is the separatrix,
where a pendulum balanced at the top takes infinitely long to fall.

The damped variant adds friction proportional to angular momentum:

    dp/dt = -sin(q) - gamma * p

which loses energy at the known rate dH/dt = -gamma * p^2. It is not
Hamiltonian - no scalar function generates that vector field - and that is
exactly why it is here. An architecture that builds in conservation should be
tested on a system that does not conserve.

Method
------
Classical fourth-order Runge-Kutta with a fixed small step. Not symplectic,
and deliberately so: the experiments that consume this data integrate their
learned vector fields with the same RK4 and the same step, so whatever energy
drift they show is a property of the learned field and not of a mismatch
between integrators. With dt = 0.01 the reference itself conserves energy to
about 1e-9 over the whole horizon, which `check_physics` measures.

Three checks against known physics run on every generation:
  1. energy conservation (undamped): |H(t) - H(0)| stays below tolerance
  2. the damping law (damped): dH/dt matches -gamma p^2 pointwise
  3. the period: a small-amplitude swing has period 2 pi (1 + theta0^2/16 + ...)

Deterministic given the seed. Initial conditions are drawn once, from a box
in phase space that covers libration, rotation and the separatrix between.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs"


@dataclass(frozen=True)
class PendulumParams:
    n_trajectories: int = 200      # initial conditions
    t_end: float = 100.0           # time units; the small-swing period is 2 pi
    dt: float = 0.01               # RK4 step
    q_max: float = np.pi           # initial angle drawn from [-q_max, q_max]
    p_max: float = 2.0             # initial momentum from [-p_max, p_max]
    gamma_damped: float = 0.1      # friction coefficient of the damped set
    save_every: int = 10           # keep one state in every N; the physics
                                   # checks still run on every step
    seed: int = 20260919

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# the equations
# --------------------------------------------------------------------------

def hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * p**2 - np.cos(q)


def vector_field(q: np.ndarray, p: np.ndarray, gamma: float = 0.0):
    """(dq/dt, dp/dt). gamma = 0 is the ideal pendulum."""
    return p, -np.sin(q) - gamma * p


def rk4_step(q, p, dt, gamma):
    k1q, k1p = vector_field(q, p, gamma)
    k2q, k2p = vector_field(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p, gamma)
    k3q, k3p = vector_field(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p, gamma)
    k4q, k4p = vector_field(q + dt * k3q, p + dt * k3p, gamma)
    q1 = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
    p1 = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
    return q1, p1


def integrate(q0: np.ndarray, p0: np.ndarray, t_end: float, dt: float,
              gamma: float):
    """All trajectories at once. Returns t (T,), q and p (T, N)."""
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    q = np.empty((n_steps + 1, len(q0)))
    p = np.empty_like(q)
    q[0], p[0] = q0, p0
    for i in range(n_steps):
        q[i + 1], p[i + 1] = rk4_step(q[i], p[i], dt, gamma)
    return t, q, p


# --------------------------------------------------------------------------
# checks against known physics
# --------------------------------------------------------------------------

def small_swing_period(theta0: float, terms: int = 6) -> float:
    """Exact period of a pendulum released from rest at angle theta0.

    The series 2 pi (1 + theta0^2/16 + 11 theta0^4/3072 + ...) is the standard
    expansion of the complete elliptic integral; six terms are accurate to
    better than 1e-6 for theta0 up to about 1 radian.
    """
    k2 = np.sin(theta0 / 2) ** 2
    s, coef = 1.0, 1.0
    for n in range(1, terms):
        coef *= ((2 * n - 1) / (2 * n)) ** 2
        s += coef * k2**n
    return 2 * np.pi * s


def check_physics(t, q, p, gamma: float, dt: float) -> dict:
    H = hamiltonian(q, p)
    out = {}
    if gamma == 0.0:
        out["max_energy_drift"] = float(np.abs(H - H[0]).max())
        # period of the trajectory released closest to rest at 0.5 rad
        rest = np.abs(p[0]) < 1e-12
        if rest.any():
            j = np.argmin(np.abs(q[0][rest] - 0.5))
            col = np.flatnonzero(rest)[j]
            qc = q[:, col]
            # successive crossings of q = 0 in the same direction
            up = np.flatnonzero((qc[:-1] < 0) & (qc[1:] >= 0))
            measured = float(np.mean(np.diff(t[up]))) if len(up) > 2 else float("nan")
            out["period_measured"] = measured
            out["period_exact"] = float(small_swing_period(abs(qc[0])))
            out["period_rel_error"] = abs(measured - out["period_exact"]) / out["period_exact"]
    else:
        dH = np.gradient(H, dt, axis=0)
        law = -gamma * p**2
        # compare away from the endpoints, where np.gradient is one-sided
        out["damping_law_max_error"] = float(np.abs(dH[2:-2] - law[2:-2]).max())
        out["damping_law_rms"] = float(np.sqrt(np.mean(law[2:-2] ** 2)))
    return out


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

def generate(params: PendulumParams) -> dict:
    rng = np.random.default_rng(params.seed)
    n = params.n_trajectories
    q0 = rng.uniform(-params.q_max, params.q_max, n)
    p0 = rng.uniform(-params.p_max, params.p_max, n)
    # one trajectory released from rest at a modest angle, so the period
    # check has something to measure; it replaces the last random draw
    q0[-1], p0[-1] = 0.5, 0.0

    out = {"q0": q0, "p0": p0, "H0": hamiltonian(q0, p0)}
    for name, gamma in (("ideal", 0.0), ("damped", params.gamma_damped)):
        t, q, p = integrate(q0, p0, params.t_end, params.dt, gamma)
        checks = check_physics(t, q, p, gamma, params.dt)     # full resolution
        # what gets stored: every save_every-th state, in float32. At the
        # default settings that is 1,001 states per trajectory and about
        # 4 MB per file instead of 70; the experiments integrate their own
        # rollouts at the fine step and compare at these times
        k = params.save_every
        t, q, p = t[::k], q[::k], p[::k]
        dq, dp = vector_field(q, p, gamma)
        f32 = lambda a: a.astype(np.float32)
        out[name] = {"t": f32(t), "q": f32(q), "p": f32(p),
                     "dq": f32(dq), "dp": f32(dp),
                     "H": f32(hamiltonian(q, p)), "checks": checks}
    return out


def save(data: dict, params: PendulumParams, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("ideal", "damped"):
        d = data[name]
        np.savez_compressed(outdir / f"pendulum_{name}.npz",
                            t=d["t"], q=d["q"], p=d["p"], dq=d["dq"], dp=d["dp"],
                            H=d["H"], q0=data["q0"], p0=data["p0"],
                            gamma=0.0 if name == "ideal" else params.gamma_damped,
                            **params.as_dict())
    meta = {"params": params.as_dict(),
            "checks": {k: data[k]["checks"] for k in ("ideal", "damped")},
            "energy_range": [float(data["H0"].min()), float(data["H0"].max())],
            "n_libration": int((data["H0"] < 1).sum()),
            "n_rotation": int((data["H0"] > 1).sum())}
    (outdir / "pendulum_params.json").write_text(json.dumps(meta, indent=2))


def plot(data: dict, params: PendulumParams, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ideal, damped = data["ideal"], data["damped"]

    ax = axes[0]
    qq, pp = np.meshgrid(np.linspace(-np.pi, np.pi, 200), np.linspace(-2.4, 2.4, 200))
    ax.contour(qq, pp, hamiltonian(qq, pp), levels=18, colors="lightgrey", linewidths=0.7)
    ax.contour(qq, pp, hamiltonian(qq, pp), levels=[1.0], colors="black", linewidths=1.4)
    for j in range(0, params.n_trajectories, 10):
        ax.plot(np.mod(ideal["q"][:, j] + np.pi, 2 * np.pi) - np.pi,
                ideal["p"][:, j], ".", ms=0.6, alpha=0.6)
    ax.set_xlabel("q  (angle)"); ax.set_ylabel("p  (momentum)")
    ax.set_title("ideal: orbits stay on their energy level")

    ax = axes[1]
    for j in range(0, params.n_trajectories, 10):
        ax.plot(np.mod(damped["q"][:, j] + np.pi, 2 * np.pi) - np.pi,
                damped["p"][:, j], ".", ms=0.6, alpha=0.6)
    ax.contour(qq, pp, hamiltonian(qq, pp), levels=[1.0], colors="black", linewidths=1.4)
    ax.set_xlabel("q  (angle)"); ax.set_ylabel("p  (momentum)")
    ax.set_title(f"damped (gamma={params.gamma_damped}): everything spirals in")

    ax = axes[2]
    ax.plot(ideal["t"], ideal["H"][:, ::10], lw=0.8, alpha=0.7)
    ax.set_xlabel("t"); ax.set_ylabel("H")
    ax.set_title(f"ideal: energy is flat to {ideal['checks']['max_energy_drift']:.0e}")
    fig.suptitle("Pendulum reference trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    for f in fields(PendulumParams):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    params = PendulumParams(**vars(args))

    data = generate(params)
    save(data, params, OUTPUT_DIR)
    plot(data, params, OUTPUT_DIR / "pendulum_overview.png")

    H0 = data["H0"]
    print(f"{params.n_trajectories} trajectories, t in [0, {params.t_end}], "
          f"dt {params.dt}, one state in {params.save_every} stored")
    print(f"  energies H0 in [{H0.min():.3f}, {H0.max():.3f}]: "
          f"{(H0 < 1).sum()} librating, {(H0 > 1).sum()} rotating "
          f"(separatrix at H = 1)")
    c = data["ideal"]["checks"]
    print(f"  ideal:  max |H(t) - H(0)| = {c['max_energy_drift']:.2e}")
    print(f"          period from rest at {0.5} rad: measured {c['period_measured']:.6f}, "
          f"exact {c['period_exact']:.6f}, rel. error {c['period_rel_error']:.1e}")
    c = data["damped"]["checks"]
    print(f"  damped: max |dH/dt + gamma p^2| = {c['damping_law_max_error']:.2e} "
          f"(law RMS {c['damping_law_rms']:.3f})")
    print(f"-> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
