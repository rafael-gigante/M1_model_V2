"""
Run Validation Plots
====================
Loads the combined spike CSVs produced by :meth:`M1Network.save_data` and
generates all post-simulation validation figures.

Run from the project root
-------------------------
    python validation/run_plots.py

Figures are written to the ``figures/`` directory.
Post-simulation analysis covers the last ``WINDOW_MS`` ms of the simulation.
"""

import os
import sys

# Add ``code/`` to the path so ``parameters`` can be imported directly.
# ``validation/`` is added automatically by Python when running this script,
# so ``plots`` and ``measures`` are already findable.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import pandas as pd
import parameters as params
import plots as pl


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    DATA_DIR     = "data"
    SIM_TIME     = params.simulation_time   # ms
    TRANSIENT_MS = 50.0                     # ms to discard from the start
    LAYER_ORDER  = params.layers_name       # ['2/3E', '2/3I', '4E', ...]
    LAYER_COLORS = pl.LAYER_COLORS

    # ------------------------------------------------------------------
    # Load spike data
    # ------------------------------------------------------------------
    spikes = {}
    for layer_name in LAYER_ORDER:
        label = layer_name.replace("/", "")
        path = os.path.join(DATA_DIR, f"combined_spikes_{label}.csv")
        df = pd.read_csv(path)
        spikes[layer_name] = df[df["time_ms"] > TRANSIENT_MS].reset_index(drop=True)
        print(f"  Loaded '{label}': {len(spikes[layer_name])} spikes (after {TRANSIENT_MS} ms transient)")

    print()

    conn_table = pd.read_csv("code/connection_table.csv", delimiter=" ", index_col=False)


    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("Generating validation figures…")

    pl.matrix_connectivity_nsyn(params.layers_name, params.n_layer, params.num_cols, conn_table)
    print("  ✓ matrix_connectivity_nsyn")

    pl.raster_plot(spikes, SIM_TIME, LAYER_ORDER, LAYER_COLORS)
    print("  ✓ raster_plot")

    pl.plot_rates(spikes, LAYER_ORDER, SIM_TIME)
    print("  ✓ plot_rates")

    pl.CV_boxplot(spikes, LAYER_ORDER, method="CV")
    print("  ✓ CV_boxplot (CV)")

    pl.CV_boxplot(spikes, LAYER_ORDER, method="CV2")
    print("  ✓ CV_boxplot (CV2)")

    pl.FR_boxplot(spikes, LAYER_ORDER)
    print("  ✓ FR_boxplot")

    pl.kuramoto_order_boxplot(spikes, LAYER_ORDER, SIM_TIME)
    print("  ✓ kuramoto_order_boxplot")

    pl.sttc_boxplot(spikes, LAYER_ORDER, SIM_TIME)
    print("  ✓ sttc_boxplot")

    pl.plot_psd(spikes, LAYER_ORDER, SIM_TIME)
    print("  ✓ plot_psd")

    print("\nAll figures saved to figures/")
