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
    plot_change_of_basis,
    plot_composition,
    plot_determinant,
    plot_dot_product,
    plot_eigenvectors,
    plot_homogeneous_translation,
    plot_letter_n,
    plot_rotation,
    plot_span_and_combinations,
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
    "custom": "Custom matrix A",
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


def matrix_for_transform(
    transform: str,
    angle_deg: float,
    custom_matrix: np.ndarray | None = None,
) -> np.ndarray:
    if transform == "custom" and custom_matrix is not None:
        return custom_matrix
    matrices = {
        "original": np.eye(2),
        "shear-fixed-x": shear_fixed_x_matrix(angle_deg),
        "shear-fixed-y": shear_fixed_y_matrix(angle_deg),
        "rotate": rotation_matrix(angle_deg),
        "shear-scaled-y": shear_scaled_y_matrix(angle_deg),
        "custom": np.eye(2),
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

        custom_matrix: np.ndarray | None = None
        if transform == "custom":
            st.markdown("**Matrix A entries**")
            st.caption("A = [[a, b], [c, d]]")
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("a (row 0, col 0)", value=1.0, step=0.1)
                c = st.number_input("c (row 1, col 0)", value=0.0, step=0.1)
            with col2:
                b = st.number_input("b (row 0, col 1)", value=0.5, step=0.1)
                d = st.number_input("d (row 1, col 1)", value=1.0, step=0.1)
            custom_matrix = np.array([[a, b], [c, d]], dtype=float)
            angle = 0.0
        else:
            angle = st.slider("Angle", -75.0, 75.0, 30.0, 1.0)

        show_original_points = st.checkbox("Show original points table", value=False)

    matrix = matrix_for_transform(transform, angle, custom_matrix)
    with output:
        fig = plot_letter_n(transform, angle, custom_matrix=custom_matrix)
        st.pyplot(fig, clear_figure=True)

    details, points = st.columns(2)
    with details:
        st.markdown("##### Transformation matrix A")
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


def render_span() -> None:
    st.subheader("Linear Combinations & Span")
    st.caption("Videos 1-2 — Vectors, linear combinations, span, and basis vectors")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        st.markdown("**Vector v₁**")
        v1x = st.number_input("v₁ x", value=2.0, step=0.5, key="span_v1x")
        v1y = st.number_input("v₁ y", value=0.0, step=0.5, key="span_v1y")
        st.markdown("**Vector v₂**")
        v2x = st.number_input("v₂ x", value=0.0, step=0.5, key="span_v2x")
        v2y = st.number_input("v₂ y", value=2.0, step=0.5, key="span_v2y")
        st.markdown("**Linear combination  a·v₁ + b·v₂**")
        a = st.slider("Scalar a", -3.0, 3.0, 1.0, 0.1, key="span_a")
        b = st.slider("Scalar b", -3.0, 3.0, 1.0, 0.1, key="span_b")

    with output:
        fig = plot_span_and_combinations((v1x, v1y), (v2x, v2y), scalar1=a, scalar2=b)
        st.pyplot(fig, clear_figure=True)

    v1 = np.array([v1x, v1y])
    v2 = np.array([v2x, v2y])
    combo = a * v1 + b * v2
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Vectors")
        st.code(f"v₁ = {format_array(v1)}\nv₂ = {format_array(v2)}", language="text")
    with col2:
        st.markdown("##### Linear combination")
        st.code(
            f"a={a:.2f}, b={b:.2f}\na·v₁ + b·v₂ = {format_array(combo)}",
            language="text",
        )
    with col3:
        st.markdown("##### Span")
        if abs(cross) < 1e-9:
            st.warning("Linearly dependent — span is a line (not all of ℝ²).")
        else:
            st.success("Linearly independent — span covers all of ℝ².")
        st.metric("2D cross product v₁×v₂", f"{cross:.4f}")


def render_determinant() -> None:
    st.subheader("The Determinant")
    st.caption("Video 6 — The determinant as signed area scaling")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        preset = st.selectbox("Matrix preset", options=list(MATRIX_PRESETS), index=0, key="det_preset")
        if preset == "Custom":
            a = st.number_input("a", value=1.0, step=0.1, key="det_a")
            b = st.number_input("b", value=0.0, step=0.1, key="det_b")
            c = st.number_input("c", value=0.0, step=0.1, key="det_c")
            d = st.number_input("d", value=1.0, step=0.1, key="det_d")
            matrix = np.array([[a, b], [c, d]], dtype=float)
        elif preset == "Rotation":
            angle = st.slider("Rotation angle", -180.0, 180.0, 45.0, 1.0, key="det_angle")
            matrix = rotation_matrix(angle)
        else:
            matrix = MATRIX_PRESETS[preset]

    with output:
        fig = plot_determinant(matrix)
        st.pyplot(fig, clear_figure=True)

    det = np.linalg.det(matrix)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### Matrix A")
        show_matrix("A", matrix)
    with col2:
        st.markdown("##### Determinant")
        st.metric("det(A)", f"{det:.4f}")
        st.metric("Area scaling factor |det|", f"{abs(det):.4f}")
    with col3:
        st.markdown("##### What it means")
        if abs(det) < 1e-9:
            st.error("det = 0 — singular matrix. Space is squished to a lower dimension. No inverse exists.")
        elif det < 0:
            st.warning(f"det < 0 — orientation flipped (mirror image). Area scaled by {abs(det):.3f}×.")
        else:
            st.success(f"det > 0 — no orientation flip. Area scaled by {det:.3f}×.")


def render_dot_product() -> None:
    st.subheader("Dot Products & Duality")
    st.caption("Video 9 — Dot product as projection, duality between vectors and linear maps")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        st.markdown("**Vector v**")
        vx = st.number_input("v x", value=3.0, step=0.5, key="dot_vx")
        vy = st.number_input("v y", value=1.0, step=0.5, key="dot_vy")
        st.markdown("**Vector w**")
        wx = st.number_input("w x", value=1.0, step=0.5, key="dot_wx")
        wy = st.number_input("w y", value=2.0, step=0.5, key="dot_wy")

    with output:
        fig = plot_dot_product((vx, vy), (wx, wy))
        st.pyplot(fig, clear_figure=True)

    v = np.array([vx, vy])
    w = np.array([wx, wy])
    dot = float(np.dot(v, w))
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Values")
        st.code(
            f"v = {format_array(v)}\nw = {format_array(w)}\nv · w = {dot:.4f}",
            language="text",
        )
    with col2:
        st.markdown("##### Geometric interpretation")
        if nv > 1e-9 and nw > 1e-9:
            cos_t = float(np.clip(dot / (nv * nw), -1.0, 1.0))
            theta = np.degrees(np.arccos(cos_t))
            proj = dot / nv
            st.code(
                f"|v| = {nv:.4f},  |w| = {nw:.4f}\n"
                f"angle θ = {theta:.2f}°,  cos θ = {cos_t:.4f}\n"
                f"proj of w onto v = {proj:.4f}",
                language="text",
            )
        if abs(dot) < 1e-9:
            st.info("v · w ≈ 0 — vectors are orthogonal (perpendicular).")
        elif dot > 0:
            st.success("v · w > 0 — vectors point in roughly the same direction.")
        else:
            st.warning("v · w < 0 — vectors point in roughly opposite directions.")


def render_eigenvectors() -> None:
    st.subheader("Eigenvectors & Eigenvalues")
    st.caption("Videos 14-15 — Eigenvectors stay on their span; they only scale by λ")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        preset = st.selectbox("Matrix preset", options=list(MATRIX_PRESETS), index=0, key="eig_preset")
        if preset == "Custom":
            a = st.number_input("a", value=3.0, step=0.1, key="eig_a")
            b = st.number_input("b", value=1.0, step=0.1, key="eig_b")
            c = st.number_input("c", value=0.0, step=0.1, key="eig_c")
            d = st.number_input("d", value=2.0, step=0.1, key="eig_d")
            matrix = np.array([[a, b], [c, d]], dtype=float)
        elif preset == "Rotation":
            angle = st.slider("Rotation angle", -180.0, 180.0, 45.0, 1.0, key="eig_angle")
            matrix = rotation_matrix(angle)
        else:
            matrix = MATRIX_PRESETS[preset]

    with output:
        fig = plot_eigenvectors(matrix)
        st.pyplot(fig, clear_figure=True)

    eigenvalues, evecs = np.linalg.eig(matrix)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Matrix A")
        show_matrix("A", matrix)
        st.markdown("##### Characteristic polynomial")
        tr = np.trace(matrix)
        det = np.linalg.det(matrix)
        st.code(f"λ² - {tr:.3f}λ + {det:.3f} = 0", language="text")
    with col2:
        st.markdown("##### Eigendecomposition  A·v = λ·v")
        rows = []
        for i, lam in enumerate(eigenvalues):
            ev = evecs[:, i]
            lam_str = f"{lam.real:.4f}" + (f" + {lam.imag:.4f}i" if abs(lam.imag) > 1e-9 else "")
            rows.append({
                "eigenvalue λ": lam_str,
                "eigenvector": f"[{ev[0].real:.3f}, {ev[1].real:.3f}]",
                "complex?": "yes" if abs(lam.imag) > 1e-9 else "no",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
        all_complex = all(abs(lam.imag) > 1e-9 for lam in eigenvalues)
        if all_complex:
            st.info("All eigenvalues are complex — no real eigenvectors (pure rotation has none).")


def render_change_of_basis() -> None:
    st.subheader("Change of Basis")
    st.caption("Video 13 — Same vector, different coordinate languages")
    controls, output = st.columns([0.34, 0.66], gap="large")

    with controls:
        st.markdown("**New basis vectors (columns of B)**")
        col1, col2 = st.columns(2)
        with col1:
            b1x = st.number_input("b₁ x", value=1.0, step=0.25, key="cob_b1x")
            b1y = st.number_input("b₁ y", value=1.0, step=0.25, key="cob_b1y")
        with col2:
            b2x = st.number_input("b₂ x", value=-1.0, step=0.25, key="cob_b2x")
            b2y = st.number_input("b₂ y", value=1.0, step=0.25, key="cob_b2y")
        st.markdown("**Vector v (standard coordinates)**")
        vx = st.number_input("v x", value=2.0, step=0.5, key="cob_vx")
        vy = st.number_input("v y", value=1.0, step=0.5, key="cob_vy")

    basis = np.array([[b1x, b2x], [b1y, b2y]], dtype=float)
    v = np.array([vx, vy], dtype=float)

    with output:
        fig = plot_change_of_basis(basis, (vx, vy))
        st.pyplot(fig, clear_figure=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Basis matrix B = [b₁ | b₂]")
        show_matrix("B", basis)
    with col2:
        st.markdown("##### Coordinate translation")
        try:
            coords_new = np.linalg.solve(basis, v)
            st.code(
                f"Standard coords:  {format_array(v)}\n"
                f"In new basis:     {format_array(coords_new)}\n\n"
                f"Verify: B @ coords_new\n= {format_array(basis @ coords_new)}",
                language="text",
            )
        except np.linalg.LinAlgError:
            st.error("Basis vectors are linearly dependent — not a valid basis.")


def render_composition() -> None:
    st.subheader("Matrix Multiplication as Composition")
    st.caption("Video 4 — Applying B first, then A, equals the single matrix A @ B")
    st.info("Read right to left: (AB)v means apply B to v first, then apply A to the result.")

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("**Matrix B** (applied first)")
        preset_b = st.selectbox("B preset", options=list(MATRIX_PRESETS), index=1, key="comp_b_preset")
        if preset_b == "Custom":
            mat_b = np.array([
                [st.number_input("B: a", value=1.0, step=0.1, key="comp_b11"),
                 st.number_input("B: b", value=0.0, step=0.1, key="comp_b12")],
                [st.number_input("B: c", value=0.0, step=0.1, key="comp_b21"),
                 st.number_input("B: d", value=1.0, step=0.1, key="comp_b22")],
            ], dtype=float)
        elif preset_b == "Rotation":
            angle_b = st.slider("B angle", -180.0, 180.0, 45.0, 1.0, key="comp_angle_b")
            mat_b = rotation_matrix(angle_b)
        else:
            mat_b = MATRIX_PRESETS[preset_b]

    with col_a:
        st.markdown("**Matrix A** (applied second)")
        preset_a = st.selectbox("A preset", options=list(MATRIX_PRESETS), index=0, key="comp_a_preset")
        if preset_a == "Custom":
            mat_a = np.array([
                [st.number_input("A: a", value=1.0, step=0.1, key="comp_a11"),
                 st.number_input("A: b", value=0.0, step=0.1, key="comp_a12")],
                [st.number_input("A: c", value=0.0, step=0.1, key="comp_a21"),
                 st.number_input("A: d", value=1.0, step=0.1, key="comp_a22")],
            ], dtype=float)
        elif preset_a == "Rotation":
            angle_a = st.slider("A angle", -180.0, 180.0, 90.0, 1.0, key="comp_angle_a")
            mat_a = rotation_matrix(angle_a)
        else:
            mat_a = MATRIX_PRESETS[preset_a]

    fig = plot_composition(mat_a, mat_b)
    st.pyplot(fig, clear_figure=True)

    ab = mat_a @ mat_b
    mc, ma, mb = st.columns(3)
    with mb:
        show_matrix("B  (first)", mat_b)
    with ma:
        show_matrix("A  (second)", mat_a)
    with mc:
        show_matrix("A @ B  (composed)", ab)
    st.caption(
        f"det(A)={np.linalg.det(mat_a):.3f}  ×  det(B)={np.linalg.det(mat_b):.3f}"
        f"  =  det(AB)={np.linalg.det(ab):.3f}  — determinants multiply!"
    )


def main() -> None:
    st.set_page_config(
        page_title="LA Visualizer",
        layout="wide",
    )
    st.title("LA Visualizer")
    st.caption(
        "Interactive linear algebra visuals"
    )

    tabs = st.tabs([
        "Transformation Lab",   # videos 3, 5
        "Vector Rotation",      # video 1 (rotation)
        "Span & Combinations",  # videos 1, 2
        "Determinant",          # video 6
        "Dot Product",          # video 9
        "Eigenvectors",         # videos 14-15
        "Change of Basis",      # video 13
        "Composition",          # video 4
        "Letter N",             # video 3
        "Homogeneous",          # bonus
    ])
    with tabs[0]:
        render_transform_lab()
    with tabs[1]:
        render_rotation()
    with tabs[2]:
        render_span()
    with tabs[3]:
        render_determinant()
    with tabs[4]:
        render_dot_product()
    with tabs[5]:
        render_eigenvectors()
    with tabs[6]:
        render_change_of_basis()
    with tabs[7]:
        render_composition()
    with tabs[8]:
        render_letter_n()
    with tabs[9]:
        render_homogeneous()


if __name__ == "__main__":
    main()
