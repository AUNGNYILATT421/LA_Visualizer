from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def rotation_matrix(theta_deg: float) -> FloatArray:
    """Return the 2D rotation matrix for an angle in degrees."""
    theta = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=float,
    )


def rotate(vector: ArrayLike, theta_deg: float) -> FloatArray:
    """Rotate a 2D vector by an angle in degrees."""
    return rotation_matrix(theta_deg) @ np.asarray(vector, dtype=float).reshape(2)


def letter_n_points(*, closed: bool = True) -> FloatArray:
    """Return the notebook's letter N control points as a 2 x N array."""
    points = np.array(
        [
            [0, 0.5, 0.5, 6, 6, 5.5, 5.5, 0],
            [0, 0, 6.42, 0, 8, 8, 1.58, 8],
        ],
        dtype=float,
    )
    return close_shape(points) if closed else points


def close_shape(points: ArrayLike) -> FloatArray:
    """Append the first point to close a 2D or 3D polyline."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must be a 2D array")
    return np.append(array, array[:, :1], axis=1)


def shear_fixed_x_matrix(theta_deg: float) -> FloatArray:
    """Return the notebook's first shear matrix."""
    theta = np.deg2rad(theta_deg)
    return np.array(
        [
            [np.cos(-theta), 0],
            [np.sin(-theta), 1],
        ],
        dtype=float,
    )


def shear_fixed_y_matrix(theta_deg: float) -> FloatArray:
    """Return the notebook's second shear matrix."""
    theta = np.deg2rad(theta_deg)
    return np.array(
        [
            [1, np.sin(theta)],
            [0, np.cos(theta)],
        ],
        dtype=float,
    )


def shear_scaled_y_matrix(theta_deg: float, y_scale: float = 1.1) -> FloatArray:
    """Return the notebook's shear matrix with increased vertical length."""
    theta = np.deg2rad(theta_deg)
    return np.array(
        [
            [1, np.sin(theta)],
            [0, y_scale],
        ],
        dtype=float,
    )


def translation_matrix(dx: float, dy: float) -> FloatArray:
    """Return a 2D homogeneous-coordinate translation matrix."""
    return np.array(
        [
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1],
        ],
        dtype=float,
    )


def to_homogeneous(points: ArrayLike, *, w: float = 1.0) -> FloatArray:
    """Convert 2D points to homogeneous 3D coordinates."""
    array = np.asarray(points, dtype=float)
    if array.shape[0] != 2:
        raise ValueError("points must have shape 2 x N")
    return np.vstack([array, np.full(array.shape[1], w, dtype=float)])
