# FASTTRANSFER

FASTTRANSFER is currently trimmed to run the Gondelach vs cylindrical B-spline comparison scenario on 20-day departure/TOF grids. The optimizer modules are still retained in `optimizer/` in case they are needed again later.

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
│   ├── optimization_Bspline_freetf.py
│   ├── optimization_*.py
│   ├── orbit_utils.py
│   └── targets.py
├── utils/
└── output/
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core runtime dependencies are NumPy, SciPy, CasADi, and Matplotlib.

## Retained Optimizers

The direct and indirect optimizer modules remain under `optimizer/`, but their root-level runnable scripts and generated result folders were cleared while this workspace is focused on the 20-day comparison scenario.
