import cortex as cx
import numpy as np
import nest
import pandas as pd
import parameters as params
import plots as pl

neuron_type = 'LIF'  # Leaky Integrate-and-Fire neuron model
connection_type = 'random'

if __name__ == "__main__":
    # Nest kernel initialization
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})  # Set the simulation resolution
    nest.total_num_virtual_procs = 5  # Set the number of virtual processes for parallel simulation
    nest.rng_seed = 67  # Set the random seed for reproducibility
    nest.overwrite_files = True
    nest.data_path = "data"  # Folder must exist
    nest.data_prefix = "m1_"

    # Create neurons and neuron groups
    print("Creating Cortex Model")
    # Calculate spatial locations for neurons in the cortical column
    spatial_locations = cx.spatial_location(params.n_layer, params.layers_name, params.num_cols, 300, 300, 2300, 50)
    
    # Plot spatial distribution of neurons in each layer
    pl.spatial_3d_plot(spatial_locations, params.layers_name)

    cortex = {}
    for layer_name in params.layers_name:
        points = np.array(spatial_locations[layer_name]).tolist()
        spatial_dist = nest.spatial.free(points)
        if neuron_type == 'LIF':
            neurons = nest.Create('iaf_psc_exp', params=params.neuron_params_LIF, positions=spatial_dist)
        cortex[layer_name] = neurons
        print(f"Created layer '{layer_name}' with {len(spatial_locations[layer_name])} neurons.")

    # Create synapses
    con_tab = pd.read_csv('code/connection_table.csv', delimiter=' ', index_col=False)
    cx.build_nest_synapses(cortex, con_tab, connection_type)
    #pl.matrix_connectivity_nsyn(params.layers_name, cortex)

    # Create background noise
    cx.background_noise(cortex, params.bg_freq, params.bg_layer)
    
    # Connect spike detectors to each layer
    cx.connect_spike_detectors(cortex)

    # Run Simulation
    print(f"Simulating for {params.simulation_time} ms...")

    nest.Simulate(params.simulation_time)

    print("Simulation complete.")

    pl.data_processing()
    