from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

cache_dir = Path.cwd() / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = cache_dir / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Arc

from la_visualizer.transforms import (
    close_shape,
    letter_n_points,
    rotate,
    rotation_matrix,
    shear_fixed_x_matrix,
    shear_fixed_y_matrix,
    shear_scaled_y_matrix,
    to_homogeneous,
    translation_matrix,
)


def save_or_show(fig: Figure, output: str | Path | None, show: bool) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rotation(
    vector: tuple[float, float] = (3.0, 1.0),
    angles_deg: float | Iterable[float] = (45.0,),
    *,
    unit_circle: bool = True,
    show_angle_arc: bool = True,
    annotate: bool = True,
) -> Figure:
    """Visualize rotating a vector by one or more angles."""
    vector_array = np.asarray(vector, dtype=float).reshape(2)
    if np.ndim(angles_deg) == 0:
        angles = [float(angles_deg)]
    else:
        angles = [float(angle) for angle in angles_deg]

    rotated = [rotate(vector_array, angle) for angle in angles]
    fig, ax = plt.subplots(figsize=(7, 7))

    if unit_circle:
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), linewidth=1, alpha=0.5, label="unit circle")

    ax.axhline(0, linewidth=1, alpha=0.6)
    ax.axvline(0, linewidth=1, alpha=0.6)
    ax.plot(
        [0, vector_array[0]],
        [0, vector_array[1]],
        linewidth=3,
        label=f"v = {np.array2string(vector_array, precision=3)}",
    )

    for angle, rotated_vector in zip(angles, rotated):
        ax.plot(
            [0, rotated_vector[0]],
            [0, rotated_vector[1]],
            linestyle="--",
            linewidth=3,
            label=f"R({angle:.1f} deg) * v",
        )

    if show_angle_arc and angles:
        radius = 0.7 * max(1.0, np.linalg.norm(vector_array))
        start_deg = np.degrees(np.arctan2(vector_array[1], vector_array[0]))
        end_deg = start_deg + angles[0]
        arc = Arc(
            (0, 0),
            2 * radius,
            2 * radius,
            angle=0,
            theta1=start_deg,
            theta2=end_deg,
            linewidth=1.5,
            linestyle=":",
        )
        ax.add_patch(arc)
        midpoint = np.deg2rad((start_deg + end_deg) / 2)
        ax.text(
            radius * np.cos(midpoint),
            radius * np.sin(midpoint),
            f"{angles[0]:.1f} deg",
            ha="center",
            va="center",
        )

    all_points = np.column_stack([vector_array, *rotated])
    max_len = max(1.3 * np.linalg.norm(all_points, axis=0).max(), 1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max_len, max_len)
    ax.set_ylim(-max_len, max_len)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.set_title("Vector Rotation via Rotation Matrix")
    ax.legend(loc="upper left")

    if annotate:
        matrix = rotation_matrix(angles[0])
        matrix_text = (
            "R(theta) = [[cos(theta), -sin(theta)],\n"
            "            [sin(theta),  cos(theta)]]\n"
            f"theta = {angles[0]:.1f} deg\n"
            f"R ~=\n{np.round(matrix, 3)}"
        )
        ax.text(
            0.02,
            0.02,
            matrix_text,
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "alpha": 0.1},
        )
        norm_v = np.linalg.norm(vector_array)
        norms = ", ".join(f"{np.linalg.norm(rotated_vector):.3f}" for rotated_vector in rotated)
        ax.text(
            0.62,
            0.02,
            f"|v| = {norm_v:.3f}\n|R(theta) * v| = {norms}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "alpha": 0.1},
        )

    return fig


def plot_letter_n(
    transform: str = "original",
    angle_deg: float = 30.0,
    *,
    custom_matrix: np.ndarray | None = None,
) -> Figure:
    """Plot the notebook letter N before or after a selected transformation.

    When custom_matrix is provided it is used directly and transform/angle_deg
    are ignored (except for the plot title).
    """
    original = letter_n_points(closed=False)
    matrix_by_transform = {
        "original": np.eye(2),
        "shear-fixed-x": shear_fixed_x_matrix(angle_deg),
        "shear-fixed-y": shear_fixed_y_matrix(angle_deg),
        "rotate": rotation_matrix(angle_deg),
        "shear-scaled-y": shear_scaled_y_matrix(angle_deg),
        "custom": np.eye(2),
    }
    if transform not in matrix_by_transform:
        valid = ", ".join(sorted(matrix_by_transform))
        raise ValueError(f"unknown transform {transform!r}; choose one of: {valid}")

    active_matrix = custom_matrix if custom_matrix is not None else matrix_by_transform[transform]
    transformed = close_shape(active_matrix @ original)
    title_by_transform = {
        "original": "N",
        "shear-fixed-x": "Sheared with fixed lengths",
        "shear-fixed-y": "Sheared with fixed lengths",
        "rotate": "Rotated",
        "shear-scaled-y": "Sheared with increased lengths",
        "custom": "Custom matrix A",
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(transformed[0], transformed[1], "bo-", markersize=8)
    ax.set_xlim(-5, 14)
    ax.set_ylim(-5, 14)
    ax.set_title(title_by_transform[transform])
    ax.grid(linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_homogeneous_translation(
    translate: tuple[float, float] = (7.0, 4.0),
) -> Figure:
    """Plot the notebook homogeneous-coordinate translation example."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    base_2d = letter_n_points(closed=False)
    n_on_z0 = close_shape(np.vstack([base_2d, np.zeros(base_2d.shape[1])]))
    n_homogeneous = close_shape(to_homogeneous(base_2d, w=1.0))

    transformed = translation_matrix(*translate) @ n_homogeneous
    projected = np.vstack(
        [
            transformed[0],
            transformed[1],
            np.zeros(transformed.shape[1], dtype=float),
        ]
    )

    ax.plot(n_on_z0[0], n_on_z0[1], n_on_z0[2], "bo-", markersize=8, label="z = 0")
    ax.plot(
        n_homogeneous[0],
        n_homogeneous[1],
        n_homogeneous[2],
        "go-",
        markersize=8,
        label="homogeneous",
    )
    ax.plot(
        transformed[0],
        transformed[1],
        transformed[2],
        "ro-",
        markersize=8,
        label="translated",
    )
    ax.plot(projected[0], projected[1], projected[2], "mo-", markersize=8, label="projected")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-5, 13)
    ax.set_ylim(-5, 13)
    ax.set_zlim(0, 3)
    ax.set_title("N")
    ax.legend(loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# 3Blue1Brown "Essence of Linear Algebra" inspired visualisations
# ---------------------------------------------------------------------------

def _arrow(ax, vec: np.ndarray, color: str, label: str, *, lw: float = 2.0) -> None:
    ax.annotate(
        "",
        xy=vec,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
    )
    offset = vec * 0.12 + 0.12
    ax.text(vec[0] + offset[0], vec[1] + offset[1], label, color=color, weight="bold", fontsize=11)


def plot_span_and_combinations(
    v1: tuple[float, float],
    v2: tuple[float, float],
    *,
    scalar1: float = 1.0,
    scalar2: float = 1.0,
    extent: int = 6,
) -> Figure:
    """Vectors 1-2: show span lines and a linear combination parallelogram."""
    v1a = np.asarray(v1, dtype=float)
    v2a = np.asarray(v2, dtype=float)
    combo = scalar1 * v1a + scalar2 * v2a

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axhline(0, color="#495057", linewidth=1)
    ax.axvline(0, color="#495057", linewidth=1)
    ax.grid(True, linestyle=":", linewidth=0.8)

    t = np.linspace(-extent * 2.5, extent * 2.5, 400)
    if np.linalg.norm(v1a) > 1e-9:
        s1 = np.outer(t, v1a).T
        ax.plot(s1[0], s1[1], color="#d62828", linewidth=1.2, alpha=0.3, label="span(v₁)")
    if np.linalg.norm(v2a) > 1e-9:
        s2 = np.outer(t, v2a).T
        ax.plot(s2[0], s2[1], color="#2a9d8f", linewidth=1.2, alpha=0.3, label="span(v₂)")

    _arrow(ax, v1a, "#d62828", "v₁")
    _arrow(ax, v2a, "#2a9d8f", "v₂")

    # parallelogram for the linear combination
    pts = np.array([
        [0, 0],
        scalar1 * v1a,
        combo,
        scalar2 * v2a,
    ])
    ax.fill(pts[:, 0], pts[:, 1], color="#f77f00", alpha=0.18)
    _arrow(ax, combo, "#f77f00", f"{scalar1:.1f}v₁+{scalar2:.1f}v₂", lw=2.5)

    limit = max(3.0, np.abs([*v1a, *v2a, *combo]).max() * 1.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title("Linear Combinations & Span  (Video 2)")
    ax.legend(loc="upper left")
    return fig


def plot_determinant(matrix: np.ndarray) -> Figure:
    """Video 6: determinant as signed area scaling of the unit square."""
    det = np.linalg.det(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    pairs = [
        (np.eye(2), "Before: unit square  (area = 1)", "#89c2d9"),
        (matrix, f"After A:  det = {det:.3f}  |area| = {abs(det):.3f}", "#f4a261" if det >= 0 else "#e63946"),
    ]

    for ax, (mat, title, color) in zip(axes, pairs):
        square = close_shape(np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float))
        ts = mat @ square
        ax.fill(ts[0], ts[1], color=color, alpha=0.45, label="parallelogram")
        ax.plot(ts[0], ts[1], color=color, linewidth=2)

        ei = mat @ np.array([1.0, 0.0])
        ej = mat @ np.array([0.0, 1.0])
        ax.annotate("", xy=ei, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=2.2))
        ax.text(ei[0] + 0.07, ei[1] + 0.07, "î", color="#d62828", weight="bold", fontsize=13)
        ax.annotate("", xy=ej, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=2.2))
        ax.text(ej[0] + 0.07, ej[1] + 0.07, "ĵ", color="#2a9d8f", weight="bold", fontsize=13)

        lim = max(3.0, max(np.abs(ei).max(), np.abs(ej).max()) * 1.6)
        ax.axhline(0, color="#aaa", linewidth=0.8)
        ax.axvline(0, color="#aaa", linewidth=0.8)
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title)

    if det < -1e-9:
        fig.suptitle("⚠ Negative det — orientation is FLIPPED (mirror image)", color="#e63946", fontsize=12)
    elif abs(det) < 1e-9:
        fig.suptitle("⚠ det = 0 — space is squished to a lower dimension (singular matrix)", fontsize=12)
    return fig


def plot_dot_product(
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> Figure:
    """Video 9: dot product as projection length times magnitude."""
    v1a = np.asarray(v1, dtype=float)
    v2a = np.asarray(v2, dtype=float)
    dot = float(np.dot(v1a, v2a))
    n1 = np.linalg.norm(v1a)
    n2 = np.linalg.norm(v2a)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.grid(True, linestyle=":", linewidth=0.8)

    _arrow(ax, v1a, "#d62828", "v")
    _arrow(ax, v2a, "#2a9d8f", "w")

    if n1 > 1e-9:
        proj_scalar = dot / (n1 ** 2)
        proj = proj_scalar * v1a
        ax.plot([v2a[0], proj[0]], [v2a[1], proj[1]], color="#8d99ae", linewidth=1.5, linestyle="--")
        _arrow(ax, proj, "#f77f00", "proj")

    if n1 > 1e-9 and n2 > 1e-9:
        cos_t = float(np.clip(dot / (n1 * n2), -1.0, 1.0))
        theta_deg = np.degrees(np.arccos(cos_t))
        r = 0.5 * min(n1, n2)
        start = np.degrees(np.arctan2(v1a[1], v1a[0]))
        end_a = np.degrees(np.arctan2(v2a[1], v2a[0]))
        arc = Arc((0, 0), 2 * r, 2 * r, angle=0,
                  theta1=min(start, end_a), theta2=max(start, end_a),
                  linewidth=1.5, linestyle=":")
        ax.add_patch(arc)
        mid = np.deg2rad((start + end_a) / 2)
        ax.text(r * 1.5 * np.cos(mid), r * 1.5 * np.sin(mid),
                f"θ={theta_deg:.1f}°", ha="center", fontsize=9)

    lim = max(3.0, max(n1, n2) * 1.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(
        f"v · w = {dot:.3f}  =  |v||w|cosθ  =  {n1:.2f} × {n2:.2f} × cosθ\n"
        "Video 9 — Dot products & duality"
    )
    return fig


def plot_eigenvectors(matrix: np.ndarray) -> Figure:
    """Videos 14-15: eigenvectors stay on their span under A, only scale by lambda."""
    eigenvalues, evecs = np.linalg.eig(matrix)
    colors = ["#d62828", "#2a9d8f"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax_idx, ax in enumerate(axes):
        ax.axhline(0, color="#aaa", linewidth=0.8)
        ax.axvline(0, color="#aaa", linewidth=0.8)
        ax.grid(True, linestyle=":", linewidth=0.6)

        any_real = False
        for i, (lam, color) in enumerate(zip(eigenvalues, colors)):
            if abs(lam.imag) > 1e-6:
                continue  # skip complex eigenvalues for 2D display
            any_real = True
            ev = evecs[:, i].real
            norm = np.linalg.norm(ev)
            if norm < 1e-9:
                continue
            ev_unit = ev / norm * 2.5

            if ax_idx == 0:
                vec = ev_unit
                label = f"v₁" if i == 0 else "v₂"
            else:
                vec = matrix @ ev_unit  # = lam * ev_unit
                label = f"λ={lam.real:.2f}"

            ax.annotate("", xy=vec, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
            ax.text(vec[0] * 1.12 + 0.1, vec[1] * 1.12 + 0.1, label,
                    color=color, weight="bold", fontsize=11)

            # faint span line
            t_vals = np.linspace(-5, 5, 100)
            span_line = np.outer(t_vals, ev / norm).T
            ax.plot(span_line[0], span_line[1], color=color, linewidth=0.8, alpha=0.25)

        if not any_real:
            ax.text(0, 0, "No real eigenvectors\n(e.g. pure rotation)",
                    ha="center", va="center", fontsize=12, color="#6c757d")

        # faint transformed square for context
        sq = close_shape(np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float))
        if ax_idx == 1:
            sq = matrix @ sq
        ax.plot(sq[0], sq[1], color="#adb5bd", linewidth=1.2, linestyle="--", alpha=0.6)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title("Eigenvectors (before)" if ax_idx == 0 else "After A:  A·v = λ·v")

    fig.suptitle("Eigenvectors don't rotate — they only scale by λ   (Videos 14-15)", fontsize=12)
    return fig


def plot_change_of_basis(
    basis_matrix: np.ndarray,
    vector: tuple[float, float],
) -> Figure:
    """Video 13: same vector described in two different coordinate systems."""
    v = np.asarray(vector, dtype=float)
    b1 = basis_matrix[:, 0]
    b2 = basis_matrix[:, 1]

    try:
        coords_new = np.linalg.solve(basis_matrix, v)
    except np.linalg.LinAlgError:
        coords_new = np.zeros(2)

    lim = max(4.0, np.linalg.norm(v) * 1.6, np.abs([*b1, *b2]).max() * 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ---- left: standard basis ----
    ax = axes[0]
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.annotate("", xy=np.array([1.0, 0.0]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=2))
    ax.text(1.1, 0.12, "î", color="#d62828", weight="bold", fontsize=13)
    ax.annotate("", xy=np.array([0.0, 1.0]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=2))
    ax.text(0.12, 1.1, "ĵ", color="#2a9d8f", weight="bold", fontsize=13)
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#f77f00", lw=2.5))
    ax.text(v[0] * 1.1 + 0.15, v[1] * 1.1 + 0.15,
            f"[{v[0]:.2f}, {v[1]:.2f}]", color="#f77f00", weight="bold", fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("Standard basis  (î, ĵ)")

    # ---- right: new basis ----
    ax = axes[1]
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axvline(0, color="#aaa", linewidth=0.8)

    # draw the new-basis grid
    for k in range(-8, 9):
        t_vals = np.linspace(-lim * 3, lim * 3, 200)
        line1 = np.outer(np.ones_like(t_vals) * k, b1) + np.outer(t_vals, b2 / max(np.linalg.norm(b2), 1e-9))
        line2 = np.outer(np.ones_like(t_vals) * k, b2) + np.outer(t_vals, b1 / max(np.linalg.norm(b1), 1e-9))
        ax.plot(line1[:, 0], line1[:, 1], color="#d62828", linewidth=0.5, alpha=0.2)
        ax.plot(line2[:, 0], line2[:, 1], color="#2a9d8f", linewidth=0.5, alpha=0.2)

    ax.annotate("", xy=b1, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=2))
    ax.text(b1[0] * 1.1 + 0.1, b1[1] * 1.1 + 0.1, "b₁", color="#d62828", weight="bold", fontsize=13)
    ax.annotate("", xy=b2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=2))
    ax.text(b2[0] * 1.1 + 0.1, b2[1] * 1.1 + 0.1, "b₂", color="#2a9d8f", weight="bold", fontsize=13)
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#f77f00", lw=2.5))
    ax.text(v[0] * 1.1 + 0.15, v[1] * 1.1 + 0.15,
            f"[{coords_new[0]:.2f}, {coords_new[1]:.2f}]ₙₑʷ",
            color="#f77f00", weight="bold", fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"New basis  (b₁, b₂)  →  coords: [{coords_new[0]:.2f}, {coords_new[1]:.2f}]")

    fig.suptitle("Change of Basis — same vector, different language   (Video 13)", fontsize=12)
    return fig


def plot_composition(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    grid_extent: int = 4,
) -> Figure:
    """Video 4: matrix multiplication as sequential composition of transformations."""
    configs = [
        (np.eye(2), "Original grid", "#d0d7de", "#89c2d9"),
        (matrix_b, "Step 1: Apply B", "#b8d8be", "#2a9d8f"),
        (matrix_a @ matrix_b, "Step 2: Then apply A  =  (AB)v", "#ffd6a5", "#f77f00"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (mat, title, grid_color, sq_color) in zip(axes, configs):
        vals = np.arange(-grid_extent, grid_extent + 1)
        for v in vals:
            for line in [
                np.array([[-grid_extent, grid_extent], [v, v]], dtype=float),
                np.array([[v, v], [-grid_extent, grid_extent]], dtype=float),
            ]:
                tl = mat @ line
                ax.plot(tl[0], tl[1], color=grid_color, linewidth=0.9)

        sq = close_shape(np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float))
        ts = mat @ sq
        ax.fill(ts[0], ts[1], color=sq_color, alpha=0.4)
        ax.plot(ts[0], ts[1], color=sq_color, linewidth=2)

        ei = mat @ np.array([1.0, 0.0])
        ej = mat @ np.array([0.0, 1.0])
        ax.annotate("", xy=ei, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=2))
        ax.annotate("", xy=ej, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=2))

        corners = mat @ np.array([
            [-grid_extent, -grid_extent, grid_extent, grid_extent],
            [-grid_extent, grid_extent, -grid_extent, grid_extent],
        ], dtype=float)
        lim = max(3.0, np.abs(corners).max() * 1.2)
        ax.axhline(0, color="#495057", linewidth=0.8)
        ax.axvline(0, color="#495057", linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title)

    fig.suptitle("Matrix Multiplication = Composition of Transformations   (Video 4)", fontsize=12)
    return fig
