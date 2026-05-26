"""
Run the M1 Cortical Column Simulation
======================================
Initialises the NEST kernel, builds the network via :class:`M1Network`,
runs the simulation, and saves the spike data.

Post-simulation analysis and plotting are handled by ``run_plots.py``.
"""

import os
import sys

# Add validation/ so plots and measures are importable from here.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation"))

import nest
import parameters as params
import plots as pl
from network import M1Network


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # NEST kernel configuration
    # ------------------------------------------------------------------
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})   # time step (ms)
    nest.total_num_virtual_procs = 8            # parallel virtual processes
    nest.rng_seed = 172                          # reproducibility seed
    nest.overwrite_files = True
    nest.data_path = "data"                     # directory must exist
    nest.data_prefix = "m1_"

    # ------------------------------------------------------------------
    # Build the M1 network
    # ------------------------------------------------------------------
    net = M1Network(
        neuron_type="LIF",
        connection_type="random",
        conn_table_path="code/connection_table.csv",
        #conn_table_path="code/cortex_spatial_new.csv",

    )

    # 1. Create spatially distributed neuron populations
    net.create_nodes()

    # 2. Optional: visualise neuron positions before the heavy connect step
    # pl.spatial_3d_plot(net.spatial_locations, params.layers_name)

    # 3. Build recurrent synapses
    net.connect()

    # 4. Attach background Poisson input
    net.create_bg_input()

    # 5. Attach spike recorders
    net.create_recorders()
    net.create_current_recorders()

    # ------------------------------------------------------------------
    # Run the simulation
    # ------------------------------------------------------------------
    net.simulate()

    # ------------------------------------------------------------------
    # Post-simulation: merge per-process spike files into combined CSVs
    # ------------------------------------------------------------------
    net.save_data(data_path="data", data_prefix="m1_")
    net.save_current_data(data_path="data", data_prefix="m1_")
