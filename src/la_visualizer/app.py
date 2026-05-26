from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from la_visualizer.plots import (
    plot_homogeneous_translation,
    plot_letter_n,
    plot_rotation,
)
from la_visualizer.transforms import (
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
        page_icon="LA",
        layout="wide",
    )
    st.title("LA Visualizer")
    st.caption("Interactive rotation, shear, and homogeneous-coordinate visuals.")

    tab_rotation, tab_letter_n, tab_homogeneous = st.tabs(
        ["Vector rotation", "Letter N", "Homogeneous translation"]
    )
    with tab_rotation:
        render_rotation()
    with tab_letter_n:
        render_letter_n()
    with tab_homogeneous:
        render_homogeneous()


if __name__ == "__main__":
    main()
