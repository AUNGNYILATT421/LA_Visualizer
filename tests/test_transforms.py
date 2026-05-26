import numpy as np

from la_visualizer.transforms import (
    close_shape,
    letter_n_points,
    rotate,
    rotation_matrix,
    to_homogeneous,
    translation_matrix,
)


def test_rotation_matrix_rotates_unit_x_to_unit_y() -> None:
    np.testing.assert_allclose(rotate([1, 0], 90), [0, 1], atol=1e-12)


def test_rotation_preserves_length() -> None:
    vector = np.array([3.0, 1.0])
    rotated = rotate(vector, 45)
    assert np.linalg.norm(rotated) == np.linalg.norm(vector)


def test_rotation_matrix_is_orthonormal() -> None:
    matrix = rotation_matrix(30)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=1e-12)


def test_letter_n_points_are_closed_by_default() -> None:
    points = letter_n_points()
    assert points.shape == (2, 9)
    np.testing.assert_allclose(points[:, 0], points[:, -1])


def test_close_shape_appends_first_point() -> None:
    points = np.array([[0, 1], [2, 3]], dtype=float)
    closed = close_shape(points)
    np.testing.assert_allclose(closed, [[0, 1, 0], [2, 3, 2]])


def test_homogeneous_translation_moves_xy_and_keeps_w() -> None:
    points = to_homogeneous(np.array([[0, 1], [2, 3]], dtype=float))
    translated = translation_matrix(7, 4) @ points
    np.testing.assert_allclose(translated, [[7, 8], [6, 7], [1, 1]])
