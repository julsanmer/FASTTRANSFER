# FASTTRANSFER

FASTTRANSFER contains the Gondelach vs cylindrical B-spline comparison workflow
used by the paper, including grid campaigns, saved-solution postprocessing, and
single-transfer examples.

## Single-Transfer Notebook

[`notebooks/simple_mars_trajectory_examples.ipynb`](notebooks/simple_mars_trajectory_examples.ipynb)
solves one fixed Earth-to-Mars transfer with no departure-date or transfer-time
grid. It runs the high-order Gondelach hodographic method and a 10-control-point
quintic cylindrical B-spline, then plots the trajectory and thrust-control
history for both methods.

The notebook uses the built-in Keplerian ephemeris so it works without the
external SPICE binary kernels:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/simple_mars_trajectory_examples.ipynb
```

## Active Scenario

Run the four kept comparison cases:

```bash
python experiments/compare_gondelach_fig2_fig3_bspline10.py --case mars --dep-step 20 --tof-step 20
python experiments/compare_gondelach_fig2_fig3_bspline10.py --case tempel1 --dep-step 20 --tof-step 20
python experiments/compare_gondelach_fig2_fig3_bspline10.py --case 1989ml --dep-step 20 --tof-step 20
python experiments/compare_gondelach_fig2_fig3_bspline10.py --case mercury --dep-step 20 --tof-step 20
```

Redraw the saved high-order Gondelach vs B-spline plots without rerunning the optimizers:

```bash
python experiments/postprocess_compare_gondelach_plots.py --case mars
```

The post-processed porkchops use calendar departure date on the x-axis and transfer time in years on the y-axis.

Run an additional B-spline variant in the same kept output folder, reusing the cached Gondelach grids and leaving the existing `bspline10` files untouched:

```bash
python experiments/run_bspline_variant_analysis.py --case mars --bspline-n-ctrl 12 --bspline-degree 5 --bspline-workers 4 --progress
```

Variant artifacts include a deterministic configuration run ID, for example
`bspline_nctrl12_deg5_a1b2c3d4_attempts.csv`. This prevents runs with different
solver settings from sharing a cache. Postprocessing accepts `12:5` when only
one matching run exists, or `12:5:a1b2c3d4` to select an explicit run.

Publication Pareto plots can combine multiple B-spline variants. By default, the postprocessor includes `10:3` and `10:5` when those cached files exist:

```bash
python experiments/postprocess_compare_gondelach_plots.py --case mars --figure-format pdf --no-plot-best-profiles
```

After generating a future 40-control-point case, include it explicitly:

```bash
python experiments/postprocess_compare_gondelach_plots.py --case mars --figure-format pdf --no-plot-best-profiles --pareto-bspline-variants 10:3 10:5 40:5
```

The kept output folders are:

```text
output/compare_gondelach_mars_bspline10_dep20d_tof20d/
output/compare_gondelach_tempel1_bspline10_dep20d_tof20d/
output/compare_gondelach_1989ml_bspline10_dep20d_tof20d/
output/compare_gondelach_mercury_bspline10_dep20d_tof20d/
```

## Active Layout

```text
FASTTRANSFER/
├── experiments/
│   ├── compare_gondelach_fig2_fig3_bspline10.py
│   ├── postprocess_compare_gondelach_plots.py
│   ├── reproduce_gondelach_fig2.py
│   ├── reproduce_gondelach_fig2_bspline_cylindrical.py
│   └── reproduce_gondelach_fig3.py
├── optimizer/
│   ├── canonical_units.py
│   ├── helpers_Bspline.py
│   ├── oneill_nelder_mead.py
│   ├── optimization_Bspline_freetf.py
│   ├── orbit_utils.py
│   └── targets.py
├── notebooks/
│   └── simple_mars_trajectory_examples.ipynb
├── utils/
└── output/
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core runtime dependencies are NumPy, SciPy, CasADi, and Matplotlib.

## Optimizer Scope

The public `optimizer/` directory contains only the cylindrical B-spline
implementation and the supporting units, orbit, target, and O'Neill
Nelder-Mead modules used by the paper experiments. Exploratory Radau and
indirect optimizers are not part of the public repository.
