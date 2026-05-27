# LA Visualizer

An interactive linear algebra visualizer built with Streamlit, inspired by
the `lavisuals.ipynb` notebook.

## Features

The app has 10 interactive tabs, each tied to a core LA concept:

| Tab                 | Concept                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| Transformation Lab  | Apply any 2×2 matrix to a deforming grid, unit square, and letter N with an animated progress slider |
| Vector Rotation     | Rotate a 2D vector by one or more angles, with rotation matrix and angle arc                         |
| Span & Combinations | Visualize span lines and a parallelogram for a·v₁ + b·v₂                                             |
| Determinant         | Side-by-side before/after showing area scaling; red tint when det < 0                                |
| Dot Product         | Projection arrow, angle arc, and v·w = \|v\|\|w\|cosθ breakdown                                      |
| Eigenvectors        | Eigenvector arrows before and after A, with span lines; notes when eigenvalues are complex           |
| Change of Basis     | Same vector shown in the standard basis vs. a custom skewed basis grid                               |
| Composition         | Three-panel grid: original → after B → after A(Bv)                                                   |
| Letter N            | Shear/rotation/custom matrix A applied to the letter N shape                                         |
| Homogeneous         | 3D plot of homogeneous-coordinate translation                                                        |

The **Letter N** tab lets you pick a preset transformation (shear fixed-x/y, rotate, shear scaled-y)
or select **Custom matrix A** to enter all four 2×2 entries manually.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Usage

Launch the interactive Streamlit app:

```bash
streamlit run src/la_visualizer/app.py
```

Generate every notebook-inspired figure to disk:

```bash
la-visualizer all --output-dir outputs
```

Or run an individual visualization:

```bash
la-visualizer rotation --vector 3 1 --angles 45 --output outputs/rotation.png
la-visualizer letter-n --transform shear-fixed-x --angle 30 --output outputs/shear.png
la-visualizer homogeneous --translate 7 4 --output outputs/homogeneous.png
```

Add `--show` to open an interactive Matplotlib window instead of saving.

## Development

```bash
python3 -m pytest
```
