from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from la_visualizer.plots import (
    plot_homogeneous_translation,
    plot_letter_n,
    plot_rotation,
)
import matplotlib.pyplot as plt

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


TRANSFORM_LABELS = {
    "original": "Original",
    "shear-fixed-x": "Shear, fixed x",
    "shear-fixed-y": "Shear, fixed y",
    "rotate": "Rotate",
    "shear-scaled-y": "Shear, scaled y",
}

MATRIX_PRESETS = {
    "Rotation": rotation_matrix(45),
    "Shear": np.array([[1.0, 0.9], [0.0, 1.0]]),
    "Stretch": np.array([[1.6, 0.0], [0.0, 0.7]]),
    "Reflection": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Projection": np.array([[1.0, 0.0], [0.0, 0.0]]),
    "Custom": np.eye(2),
}


def format_array(array: np.ndarray) -> str:
    return np.array2string(array, precision=3, suppress_small=True)


def matrix_for_transform(transform: str, angle_deg: float) -> np.ndarray:
    matrices = {
        "original": np.eye(2),
        "shear-fixed-x": shear_fixed_x_matrix(angle_deg),
        "shear-fixed-y": shear_fixed_y_matrix(angle_deg),
        "rotate": rotation_matrix(angle_deg),
        "shear-scaled-y": shear_scaled_y_matrix(angle_deg),
    }
    return matrices[transform]


def show_matrix(label: str, matrix: np.ndarray) -> None:
    st.caption(label)
    st.code(format_array(matrix), language="text")


def figure_png(fig: Figure) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    return buffer.getvalue()


def interpolate_matrix(matrix: np.ndarray, progress: float) -> np.ndarray:
    return (1.0 - progress) * np.eye(2) + progress * matrix


def grid_lines(extent: int, spacing: float) -> list[np.ndarray]:
    values = np.arange(-extent, extent + spacing, spacing)
    lines = []
    for value in values:
        lines.append(np.array([[-extent, extent], [value, value]], dtype=float))
        lines.append(np.array([[value, value], [-extent, extent]], dtype=float))
    return lines


def draw_vector(ax, vector: np.ndarray, color: str, label: str, *, alpha: float = 1.0) -> None:
    ax.arrow(
        0,
        0,
        vector[0],
        vector[1],
        width=0.025,
        head_width=0.18,
        length_includes_head=True,
        color=color,
        alpha=alpha,
    )
    ax.text(vector[0] * 1.08, vector[1] * 1.08, label, color=color, weight="bold")


def plot_transform_lab(
    matrix: np.ndarray,
    *,
    progress: float,
    vector: tuple[float, float],
    grid_extent: int,
    spacing: float,
    show_original: bool,
    show_letter: bool,
) -> Figure:
    active_matrix = interpolate_matrix(matrix, progress)
    fig, ax = plt.subplots(figsize=(8, 8))

    for line in grid_lines(grid_extent, spacing):
        transformed = active_matrix @ line
        ax.plot(transformed[0], transformed[1], color="#d0d7de", linewidth=0.8, zorder=1)

    if show_original:
        for line in grid_lines(grid_extent, spacing):
            ax.plot(line[0], line[1], color="#eef1f4", linewidth=0.6, zorder=0)

    square = close_shape(np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float))
    transformed_square = active_matrix @ square
    ax.fill(
        transformed_square[0],
        transformed_square[1],
        color="#89c2d9",
        alpha=0.28,
        label="unit square",
        zorder=2,
    )
    ax.plot(transformed_square[0], transformed_square[1], color="#227c9d", linewidth=2, zorder=3)

    if show_letter:
        letter = letter_n_points(closed=True) / 2.4 - np.array([[1.2], [1.3]])
        transformed_letter = active_matrix @ letter
        ax.plot(transformed_letter[0], transformed_letter[1], color="#5a189a", linewidth=3, zorder=4)

    draw_vector(ax, active_matrix @ np.array([1.0, 0.0]), "#d62828", "i")
    draw_vector(ax, active_matrix @ np.array([0.0, 1.0]), "#2a9d8f", "j")

    source_vector = np.asarray(vector, dtype=float)
    if show_original:
        draw_vector(ax, source_vector, "#8d99ae", "v", alpha=0.55)
    draw_vector(ax, active_matrix @ source_vector, "#f77f00", "A v")

    corners = np.array(
        [
            [-grid_extent, -grid_extent, grid_extent, grid_extent, 0],
            [-grid_extent, grid_extent, -grid_extent, grid_extent, 0],
        ],
        dtype=float,
    )
    transformed_corners = active_matrix @ corners
    limit = max(3.0, np.abs(transformed_corners).max() * 1.15, np.linalg.norm(active_matrix @ source_vector) * 1.3)
    ax.axhline(0, color="#495057", linewidth=1)
    ax.axvline(0, color="#495057", linewidth=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Linear Transformation Lab")
    ax.legend(loc="upper left")
    return fig


def matrix_readout(matrix: np.ndarray) -> list[dict[str, str]]:
    eigenvalues = np.linalg.eigvals(matrix)
    return [
        {"property": "determinant", "value": f"{np.linalg.det(matrix):.3f}"},
        {"property": "trace", "value": f"{np.trace(matrix):.3f}"},
        {"property": "eigenvalue 1", "value": f"{eigenvalues[0]:.3g}"},
        {"property": "eigenvalue 2", "value": f"{eigenvalues[1]:.3g}"},
    ]


def show_vector_metrics(vector: tuple[float, float], angles: Iterable[float]) -> None:
    vector_array = np.asarray(vector, dtype=float)
    rows = []
    for angle in angles:
        rotated = rotate(vector_array, angle)
        rows.append(
            {
                "angle": f"{angle:.1f} deg",
                "x": rotated[0],
                "y": rotated[1],
                "length": np.linalg.norm(rotated),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_rotation() -> None:
    st.subheader("Vector Rotation")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        x_value = st.number_input("Vector x", value=3.0, step=0.5)
        y_value = st.number_input("Vector y", value=1.0, step=0.5)
        primary_angle = st.slider("Primary angle", -180.0, 180.0, 45.0, 1.0)
        compare_angles = st.multiselect(
            "Compare angles",
            options=[-135.0, -90.0, -45.0, 0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0],
            default=[],
        )
        unit_circle = st.checkbox("Unit circle", value=True)
        show_angle_arc = st.checkbox("Angle arc", value=True)
        annotate = st.checkbox("Annotations", value=True)

    angles = [primary_angle, *compare_angles]
    with output:
        fig = plot_rotation(
            (x_value, y_value),
            angles,
            unit_circle=unit_circle,
            show_angle_arc=show_angle_arc,
            annotate=annotate,
        )
        st.pyplot(fig, clear_figure=True)

    readout, matrix = st.columns(2)
    with readout:
        st.markdown("##### Rotated vectors")
        show_vector_metrics((x_value, y_value), angles)
    with matrix:
        st.markdown("##### Rotation matrix")
        show_matrix(f"theta = {primary_angle:.1f} deg", rotation_matrix(primary_angle))


def render_transform_lab() -> None:
    st.subheader("Transformation Lab")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        preset = st.selectbox("Matrix preset", options=list(MATRIX_PRESETS), index=0)
        if preset == "Custom":
            a = st.number_input("a", value=1.0, step=0.1)
            b = st.number_input("b", value=0.0, step=0.1)
            c = st.number_input("c", value=0.0, step=0.1)
            d = st.number_input("d", value=1.0, step=0.1)
            matrix = np.array([[a, b], [c, d]], dtype=float)
        elif preset == "Rotation":
            angle = st.slider("Rotation angle", -180.0, 180.0, 45.0, 1.0)
            matrix = rotation_matrix(angle)
        else:
            matrix = MATRIX_PRESETS[preset]

        progress = st.slider("Transformation progress", 0.0, 1.0, 1.0, 0.02)
        grid_extent = st.slider("Grid extent", 2, 10, 5, 1)
        spacing = st.select_slider("Grid spacing", options=[0.5, 1.0, 2.0], value=1.0)
        vx = st.number_input("Vector v x", value=2.0, step=0.25)
        vy = st.number_input("Vector v y", value=1.0, step=0.25)
        show_original = st.checkbox("Show original plane", value=True)
        show_letter = st.checkbox("Show letter N", value=True)

    fig = plot_transform_lab(
        matrix,
        progress=progress,
        vector=(vx, vy),
        grid_extent=grid_extent,
        spacing=spacing,
        show_original=show_original,
        show_letter=show_letter,
    )
    png = figure_png(fig)
    with output:
        st.pyplot(fig, clear_figure=True)
        st.download_button(
            "Download PNG",
            data=png,
            file_name="linear_transformation_lab.png",
            mime="image/png",
        )

    active_matrix = interpolate_matrix(matrix, progress)
    matrix_col, metrics_col, vector_col = st.columns(3)
    with matrix_col:
        st.markdown("##### Target matrix")
        show_matrix("A", matrix)
    with metrics_col:
        st.markdown("##### Matrix behavior")
        st.dataframe(matrix_readout(matrix), hide_index=True, use_container_width=True)
    with vector_col:
        st.markdown("##### Current vector")
        vector = np.array([vx, vy], dtype=float)
        transformed = active_matrix @ vector
        st.code(
            f"v = {format_array(vector)}\nA_t v = {format_array(transformed)}",
            language="text",
        )


def render_letter_n() -> None:
    st.subheader("Letter N Transformations")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        transform = st.selectbox(
            "Transformation",
            options=list(TRANSFORM_LABELS),
            format_func=TRANSFORM_LABELS.get,
            index=1,
        )
        angle = st.slider("Angle", -75.0, 75.0, 30.0, 1.0)
        show_original_points = st.checkbox("Show original points table", value=False)

    matrix = matrix_for_transform(transform, angle)
    with output:
        fig = plot_letter_n(transform, angle)
        st.pyplot(fig, clear_figure=True)

    details, points = st.columns(2)
    with details:
        st.markdown("##### Transformation matrix")
        show_matrix(TRANSFORM_LABELS[transform], matrix)
    with points:
        st.markdown("##### Shape summary")
        original = letter_n_points(closed=False)
        transformed = matrix @ original
        st.metric("Control points", original.shape[1])
        st.metric("Transformed width", f"{np.ptp(transformed[0]):.2f}")
        st.metric("Transformed height", f"{np.ptp(transformed[1]):.2f}")

    if show_original_points:
        st.dataframe(
            [{"point": index + 1, "x": x, "y": y} for index, (x, y) in enumerate(original.T)],
            hide_index=True,
            use_container_width=True,
        )


def render_homogeneous() -> None:
    st.subheader("Homogeneous Translation")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        dx = st.slider("Translate x", -10.0, 10.0, 7.0, 0.5)
        dy = st.slider("Translate y", -10.0, 10.0, 4.0, 0.5)

    with output:
        fig = plot_homogeneous_translation((dx, dy))
        st.pyplot(fig, clear_figure=True)

    matrix, sample = st.columns(2)
    transform = translation_matrix(dx, dy)
    with matrix:
        st.markdown("##### Translation matrix")
        show_matrix("homogeneous coordinates", transform)
    with sample:
        st.markdown("##### First point")
        point = to_homogeneous(letter_n_points(closed=False)[:, :1])
        translated = transform @ point
        st.code(
            f"before: {format_array(point[:, 0])}\nafter:  {format_array(translated[:, 0])}",
            language="text",
        )


def main() -> None:
    st.set_page_config(
        page_title="LA Visualizer",
        layout="wide",
    )
    st.title("LA Visualizer")
    st.caption("Interactive matrix visuals inspired by geometric linear algebra.")

    tab_lab, tab_rotation, tab_letter_n, tab_homogeneous = st.tabs(
        ["Transformation lab", "Vector rotation", "Letter N", "Homogeneous translation"]
    )
    with tab_lab:
        render_transform_lab()
    with tab_rotation:
        render_rotation()
    with tab_letter_n:
        render_letter_n()
    with tab_homogeneous:
        render_homogeneous()


if __name__ == "__main__":
    main()
