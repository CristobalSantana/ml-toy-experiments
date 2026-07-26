"""
manim_figures.py -- Professional, equation-free animations of the Phase 3
KAN vs MLP comparison, for a non-academic audience.

Reads plain numpy/json data only (outputs/phase3_results.json,
outputs/manim_data/solution_fit_dense.npz, and the generator's own
reference solution) -- no torch import here, so this can run in a
separate, lighter virtual environment than the one used to train the
models.

Four scenes:
  TrainingRace      -- both models' prediction error falling during training
  CostVsAccuracy    -- training time vs. final error, 5 runs each
  RealityCheck      -- predicted vs. real battery-charging curve, over time
  ParticleFilling   -- the ground-truth physics itself: how lithium fills
                        the electrode particle, straight from the
                        generator's reference solution (no ML involved)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *

NEAR_BLACK = "#0e0e0e"
OFFWHITE = "#f5f0e8"
MLP_BLUE = "#4fa8ff"
KAN_ORANGE = "#ff5300"
GREY_TEXT = "#9a9a9a"

DATA_DIR = Path(__file__).parent / "outputs"
with open(DATA_DIR / "phase3_results.json") as f:
    PAYLOAD = json.load(f)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_NPZ = REPO_ROOT / "generators" / "diffusion_1d" / "outputs" / "diffusion_1d_solution.npz"


def curve_arrays(payload, arch, key):
    runs = payload["results"][arch]
    epochs = np.array([h["epoch"] for h in runs[0]["history"]])
    vals = np.array([[h[key] for h in r["history"]] for r in runs])
    return epochs, vals.mean(axis=0)


class TrainingRace(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        title = Text("Learning to Solve the Physics", font_size=36, color=OFFWHITE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        subtitle = Text("Prediction error while training, averaged over 5 runs",
                        font_size=20, color=GREY_TEXT).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle), run_time=1.0)

        epochs, mlp_mean = curve_arrays(PAYLOAD, "mlp", "rel_l2")
        _, kan_mean = curve_arrays(PAYLOAD, "kan", "rel_l2")
        log_mlp, log_kan = np.log10(mlp_mean), np.log10(kan_mean)
        y_all = np.concatenate([log_mlp, log_kan])
        y_floor = y_all.min() - 0.15
        y_span = (y_all.max() + 0.15) - y_floor
        # Shift so the plotted range starts at 0 -- keeps Axes' default
        # x-axis (drawn at y=0) pinned to the BOTTOM of the chart instead
        # of floating wherever raw log10(error)=0 happens to fall.
        shifted_mlp = log_mlp - y_floor
        shifted_kan = log_kan - y_floor

        axes = Axes(
            x_range=[0, epochs.max(), epochs.max() / 6],
            y_range=[0, y_span, y_span / 5],
            x_length=10, y_length=5,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.35)

        x_label = Text("Training Progress", font_size=20, color=OFFWHITE).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Prediction error", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        mlp_points = [axes.c2p(e, l) for e, l in zip(epochs, shifted_mlp)]
        kan_points = [axes.c2p(e, l) for e, l in zip(epochs, shifted_kan)]
        mlp_curve = VMobject(color=MLP_BLUE, stroke_width=5).set_points_smoothly(mlp_points)
        kan_curve = VMobject(color=KAN_ORANGE, stroke_width=5).set_points_smoothly(kan_points)
        mlp_dot = Dot(color=MLP_BLUE, radius=0.09).move_to(mlp_points[0])
        kan_dot = Dot(color=KAN_ORANGE, radius=0.09).move_to(kan_points[0])

        self.play(
            Create(mlp_curve), Create(kan_curve),
            MoveAlongPath(mlp_dot, mlp_curve), MoveAlongPath(kan_dot, kan_curve),
            run_time=4.5, rate_func=linear,
        )

        mlp_label = Text("MLP", font_size=26, color=MLP_BLUE, weight=BOLD).next_to(mlp_points[-1], UP, buff=0.25)
        kan_label = Text("KAN", font_size=26, color=KAN_ORANGE, weight=BOLD).next_to(kan_points[-1], DOWN, buff=0.3)
        self.play(FadeIn(mlp_label, shift=DOWN * 0.1), FadeIn(kan_label, shift=UP * 0.1), run_time=0.6)
        self.wait(1.8)


class CostVsAccuracy(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        title = Text("Prediction Error vs Training Time (5 runs each)",
                    font_size=30, color=OFFWHITE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=1.0)

        mlp_runs, kan_runs = PAYLOAD["results"]["mlp"], PAYLOAD["results"]["kan"]
        mlp_times = np.array([r["wall_time"] for r in mlp_runs])
        mlp_errs = np.log10(np.array([r["final_rel_l2"] for r in mlp_runs]))
        kan_times = np.array([r["wall_time"] for r in kan_runs])
        kan_errs = np.log10(np.array([r["final_rel_l2"] for r in kan_runs]))

        x_max = max(mlp_times.max(), kan_times.max()) * 1.15
        y_all = np.concatenate([mlp_errs, kan_errs])
        y_floor = y_all.min() - 0.15
        y_span = (y_all.max() + 0.15) - y_floor
        shifted_mlp_errs = mlp_errs - y_floor
        shifted_kan_errs = kan_errs - y_floor

        axes = Axes(
            x_range=[0, x_max, x_max / 5],
            y_range=[0, y_span, y_span / 4],
            x_length=10, y_length=5,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.05)

        x_label = Text("Training Time [s]", font_size=20, color=OFFWHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Prediction error", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        mlp_dots = VGroup(*[Dot(axes.c2p(tm, e), color=MLP_BLUE, radius=0.1)
                            for tm, e in zip(mlp_times, shifted_mlp_errs)])
        kan_dots = VGroup(*[Dot(axes.c2p(tm, e), color=KAN_ORANGE, radius=0.1)
                            for tm, e in zip(kan_times, shifted_kan_errs)])

        self.play(LaggedStart(*[GrowFromCenter(d) for d in mlp_dots], lag_ratio=0.2), run_time=1.2)
        mlp_tag = Text("MLP", font_size=24, color=MLP_BLUE, weight=BOLD).next_to(mlp_dots, UP, buff=0.3)
        self.play(FadeIn(mlp_tag, shift=DOWN * 0.1), run_time=0.5)

        self.play(LaggedStart(*[GrowFromCenter(d) for d in kan_dots], lag_ratio=0.2), run_time=1.2)
        kan_tag = Text("KAN", font_size=24, color=KAN_ORANGE, weight=BOLD).next_to(kan_dots, UP, buff=0.3)
        self.play(FadeIn(kan_tag, shift=DOWN * 0.1), run_time=0.5)

        speedup = kan_times.mean() / mlp_times.mean()
        caption = Text(f"KAN: more accurate, ~{speedup:.0f}× slower to train",
                       font_size=24, color=OFFWHITE).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)


class RealityCheck(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        data = np.load(DATA_DIR / "manim_data" / "solution_fit_dense.npz")
        x_um, t_min = data["x_um"], data["t_min"]
        charge_ref, charge_mlp, charge_kan = data["charge_ref"], data["charge_mlp"], data["charge_kan"]
        n_frames = len(t_min)

        title = Text("Does It Match Reality?", font_size=36, color=OFFWHITE, weight=BOLD).to_edge(UP, buff=0.5)
        subtitle = Text("Charging a battery electrode: simulated (ground truth) vs. predicted",
                        font_size=19, color=GREY_TEXT).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle), run_time=1.0)

        axes = Axes(
            x_range=[x_um.min(), x_um.max(), (x_um.max() - x_um.min()) / 4],
            y_range=[0, 100, 25],
            x_length=10, y_length=4.3,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.6)

        left_label = Text("Charging surface", font_size=17, color=GREY_TEXT)
        left_label.next_to(axes.c2p(x_um.min(), 0), DOWN, buff=0.3)
        right_label = Text("Particle core", font_size=17, color=GREY_TEXT)
        right_label.next_to(axes.c2p(x_um.max(), 0), DOWN, buff=0.3)
        y_label = Text("Charge level", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(left_label), FadeIn(right_label), FadeIn(y_label), run_time=1.0)

        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=OFFWHITE, stroke_width=6),
                   Text("Real physics", font_size=16, color=OFFWHITE)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=MLP_BLUE, stroke_width=4),
                   Text("MLP prediction", font_size=16, color=MLP_BLUE)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=KAN_ORANGE, stroke_width=4),
                   Text("KAN prediction", font_size=16, color=KAN_ORANGE)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.7).shift(DOWN * 0.9)
        self.play(FadeIn(legend), run_time=0.8)

        time_tracker = ValueTracker(0.0)

        def interp(arr):
            idx_f = time_tracker.get_value() * (n_frames - 1)
            i0 = int(np.floor(idx_f))
            i1 = min(i0 + 1, n_frames - 1)
            w = idx_f - i0
            return arr[i0] * (1 - w) + arr[i1] * w

        def make_curve(y_values, color, width, opacity=1.0):
            points = [axes.c2p(xv, yv) for xv, yv in zip(x_um, y_values)]
            return VMobject(stroke_color=color, stroke_width=width,
                            stroke_opacity=opacity).set_points_smoothly(points)

        # The three curves nearly coincide almost everywhere, so a KAN line
        # drawn at the same width would completely hide MLP and the
        # reference underneath it. Nesting decreasing widths (thick pale
        # reference -> medium MLP -> thin KAN) keeps all three visible as
        # concentric "halo" lines, and any real deviation between them
        # shows up as one line straying outside another's band.
        ref_curve = always_redraw(lambda: make_curve(interp(charge_ref), OFFWHITE, 10, opacity=0.9))
        mlp_curve = always_redraw(lambda: make_curve(interp(charge_mlp), MLP_BLUE, 5))
        kan_curve = always_redraw(lambda: make_curve(interp(charge_kan), KAN_ORANGE, 2))
        time_label = always_redraw(lambda: Text(
            f"t = {interp(t_min):.1f} min", font_size=24, color=OFFWHITE,
        ).to_corner(UR, buff=0.6))

        self.add(ref_curve, mlp_curve, kan_curve, time_label)
        self.play(FadeIn(VGroup(ref_curve, mlp_curve, kan_curve, time_label)), run_time=0.6)
        self.play(time_tracker.animate.set_value(1.0), run_time=8.0, rate_func=linear)
        self.wait(0.4)

        caption = Text("Both models learned to reproduce the real physics.",
                       font_size=23, color=OFFWHITE).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.7)
        self.wait(1.8)


class ParticleFilling(Scene):
    """The ground-truth physics itself, straight from the finite-difference
    reference solution -- no neural networks involved. Two synchronized
    panels: a particle cross-section filling with concentration from the
    surface inward (the real 1D solution mapped radially outside-in for an
    intuitive "cross-section" visual), and the matching concentration
    profile as a line plot, so the viewer sees the curve's tilt and the
    particle's uneven fill are the same fact shown two ways.
    """

    def construct(self):
        self.camera.background_color = NEAR_BLACK

        data = np.load(REFERENCE_NPZ)
        x, t, C = data["x"], data["t"], data["C"]
        t_end = float(t.max())
        c_min, c_max = float(C.min()), float(C.max())
        x_hat_grid = x / float(x.max())   # 0 (surface) .. 1 (center)
        n_t = len(t)

        def c_at(time_frac: float) -> np.ndarray:
            """Interpolated concentration profile (nx,) at dimensionless time_frac in [0,1]."""
            idx_f = time_frac * (n_t - 1)
            i0 = int(np.floor(idx_f))
            i1 = min(i0 + 1, n_t - 1)
            w = idx_f - i0
            return C[i0] * (1 - w) + C[i1] * w

        title = Text("How Lithium Fills the Particle", font_size=34, color=OFFWHITE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        # ---------------- left panel: particle cross-section ----------------
        circle_center = np.array([-3.6, -0.5, 0])
        radius = 2.0
        img_res = 400
        LOW_RGB = np.array([55, 50, 48])     # dim, "unlit" -- low concentration
        HIGH_RGB = np.array([255, 83, 0])    # KAN_ORANGE -- high concentration
        BG_RGB = np.array([14, 14, 14])      # matches NEAR_BLACK exactly

        time_tracker = ValueTracker(0.0)     # 0..1 dimensionless time, 0..t_end
        marker_x = ValueTracker(0.0)         # 0 (surface) .. 1 (center)

        def particle_array(time_frac: float) -> np.ndarray:
            c_norm = np.clip((c_at(time_frac) - c_min) / (c_max - c_min), 0, 1)
            yy, xx = np.mgrid[0:img_res, 0:img_res]
            c0 = (img_res - 1) / 2
            r = np.sqrt((xx - c0) ** 2 + (yy - c0) ** 2) / (img_res / 2)   # 0 center .. 1 edge
            x_hat = np.clip(1 - r, 0, 1)   # edge(r=1)->surface(0); center(r=0)->center(1)
            c_pixel = np.interp(x_hat.ravel(), x_hat_grid, c_norm).reshape(img_res, img_res)
            rgb = LOW_RGB + c_pixel[..., None] * (HIGH_RGB - LOW_RGB)
            rgb = np.where(r[..., None] <= 1.0, rgb, BG_RGB)
            return rgb.astype(np.uint8)

        particle_img = always_redraw(
            lambda: ImageMobject(particle_array(time_tracker.get_value()))
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
            .scale_to_fit_width(2 * radius)
            .move_to(circle_center)
        )
        particle_outline = Circle(radius=radius, color=OFFWHITE, stroke_width=2.5).move_to(circle_center)

        left_caption = Text(
            "One electrode particle\n(a grain of solid material),\ncut in half",
            font_size=16, color=GREY_TEXT, line_spacing=0.9,
        )
        left_caption.next_to(circle_center + UP * radius, UP, buff=0.2)

        # an inward-pointing arrow at the edge, labeled, so the surface entry
        # point is self-explanatory without needing prose underneath
        entry_dir = np.array([-1.0, 0.0, 0.0])   # left edge, same axis the marker sweeps
        entry_outer = circle_center + entry_dir * (radius + 0.9)
        entry_inner = circle_center + entry_dir * (radius + 0.05)
        entry_arrow = Arrow(entry_outer, entry_inner, color=OFFWHITE, stroke_width=3,
                            max_tip_length_to_length_ratio=0.35, buff=0)
        entry_label = Text("Li⁺ enters here", font_size=16, color=OFFWHITE)
        entry_label.next_to(entry_outer, UP, buff=0.15)
        # keep it on screen: the label is wider than the arrow it sits
        # above, so centering it on entry_outer can push its left edge
        # past the frame boundary -- nudge right if so.
        safe_left_edge = -6.9
        if entry_label.get_left()[0] < safe_left_edge:
            entry_label.shift(RIGHT * (safe_left_edge - entry_label.get_left()[0]))

        center_label = Text("center", font_size=15, color=OFFWHITE)
        center_label.move_to(circle_center + DOWN * 0.4)

        # ---------------- right panel: concentration profile ----------------
        axes = Axes(
            x_range=[0, 1, 0.25], y_range=[c_min, c_max, (c_max - c_min) / 4],
            x_length=5.4, y_length=4.3,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).move_to([3.4, -0.5, 0])

        right_caption = Text("Concentration profile", font_size=18, color=GREY_TEXT)
        right_caption.next_to(axes, UP, buff=0.35)
        x_left_label = Text("Surface", font_size=16, color=GREY_TEXT).next_to(axes.c2p(0, c_min), DOWN, buff=0.25)
        x_right_label = Text("Center", font_size=16, color=GREY_TEXT).next_to(axes.c2p(1, c_min), DOWN, buff=0.25)
        y_label = Text("Concentration", font_size=18, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        curve = always_redraw(lambda: VMobject(color=KAN_ORANGE, stroke_width=4).set_points_smoothly(
            [axes.c2p(xh, cv) for xh, cv in zip(x_hat_grid, c_at(time_tracker.get_value()))]
        ))

        clock = always_redraw(lambda: Text(
            f"t = {time_tracker.get_value() * t_end / 60:.1f} min", font_size=24, color=OFFWHITE, weight=BOLD,
        ).to_corner(UR, buff=0.5))

        # ---------------- the marker tying curve position <-> radius ----------------
        def marker_right_group():
            xh = marker_x.get_value()
            cv = float(np.interp(xh, x_hat_grid, c_at(time_tracker.get_value())))
            line = DashedLine(axes.c2p(xh, c_min), axes.c2p(xh, cv), color=OFFWHITE,
                              stroke_width=2, dash_length=0.08)
            dot = Dot(axes.c2p(xh, cv), color=OFFWHITE, radius=0.07)
            return VGroup(line, dot)

        def marker_left_dot():
            xh = marker_x.get_value()
            pos = circle_center + LEFT * radius * (1 - xh)
            return Dot(pos, color=OFFWHITE, radius=0.09, stroke_color=NEAR_BLACK, stroke_width=1.5)

        marker_right_mob = always_redraw(marker_right_group)
        marker_left_mob = always_redraw(marker_left_dot)

        # ---------------- sequence ----------------
        self.play(
            FadeIn(particle_outline), FadeIn(particle_img),
            FadeIn(left_caption), FadeIn(center_label),
            Create(axes), FadeIn(right_caption), FadeIn(x_left_label), FadeIn(x_right_label), FadeIn(y_label),
            FadeIn(clock),
            run_time=1.3,
        )
        self.play(GrowArrow(entry_arrow), FadeIn(entry_label, shift=RIGHT * 0.15), run_time=0.6)
        self.add(curve, marker_right_mob, marker_left_mob)
        self.play(FadeIn(curve), FadeIn(marker_right_mob), FadeIn(marker_left_mob), run_time=0.3)

        # a full surface-to-center sweep, at t=0, to teach the correspondence
        # between "position on the curve" and "radius in the particle" ...
        self.play(marker_x.animate.set_value(1.0), run_time=1.2, rate_func=smooth)
        # ... then settle to a resting position that stays visible as an
        # anchor throughout the main event below.
        self.play(marker_x.animate.set_value(0.35), run_time=0.4, rate_func=smooth)

        # the main event: 0 -> 20 minutes, both panels driven by the same clock
        self.play(time_tracker.animate.set_value(1.0), run_time=6.3, rate_func=linear)
        self.wait(0.5)

        caption = Text("The tilt in the curve IS the uneven filling of the particle.",
                       font_size=22, color=OFFWHITE).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.6)
        self.wait(0.4)
