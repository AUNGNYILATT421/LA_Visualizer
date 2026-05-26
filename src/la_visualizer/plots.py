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


def plot_letter_n(transform: str = "original", angle_deg: float = 30.0) -> Figure:
    """Plot the notebook letter N before or after a selected transformation."""
    original = letter_n_points(closed=False)
    matrix_by_transform = {
        "original": np.eye(2),
        "shear-fixed-x": shear_fixed_x_matrix(angle_deg),
        "shear-fixed-y": shear_fixed_y_matrix(angle_deg),
        "rotate": rotation_matrix(angle_deg),
        "shear-scaled-y": shear_scaled_y_matrix(angle_deg),
    }
    if transform not in matrix_by_transform:
        valid = ", ".join(sorted(matrix_by_transform))
        raise ValueError(f"unknown transform {transform!r}; choose one of: {valid}")

    transformed = close_shape(matrix_by_transform[transform] @ original)
    title_by_transform = {
        "original": "N",
        "shear-fixed-x": "Sheared with fixed lengths",
        "shear-fixed-y": "Sheared with fixed lengths",
        "rotate": "Rotated",
        "shear-scaled-y": "Sheared with increased lengths",
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
