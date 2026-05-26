# LA Visualizer

A small Python project built from the supplied `lavisuals.ipynb` notebook. It visualizes:

- 2D vector rotation with rotation matrices
- Shear and rotation transformations on a letter `N`
- Homogeneous-coordinate translation in 3D

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

Generate every notebook-inspired figure:

```bash
la-visualizer all --output-dir outputs
```

Or run an individual visualization:

```bash
la-visualizer rotation --vector 3 1 --angles 45 --output outputs/rotation.png
la-visualizer letter-n --transform shear-fixed-x --angle 30 --output outputs/shear.png
la-visualizer homogeneous --translate 7 4 --output outputs/homogeneous.png
```

Each command writes a PNG file. Add `--show` to open an interactive Matplotlib window.

## Development

```bash
python3 -m pytest
```
