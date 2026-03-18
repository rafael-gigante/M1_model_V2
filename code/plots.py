import matplotlib.pyplot as plt
import nest
import glob
import os
import seaborn as sns
import numpy as np
import pandas as pd
import parameters as params
from scipy.ndimage import gaussian_filter1d

def data_processing(data_path="data", data_prefix="m1_", layers=['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']):
    for layer in layers:
        # Find all spike files for this run
        # NEST usually names them: <prefix>spikes-<GID>-<ProcessID>.dat (or .csv)
        search_pattern = os.path.join(data_path, f"{data_prefix}spikes_{layer}-*.dat")
        spike_files = glob.glob(search_pattern)

        if not spike_files:
            print(f"No spike files found for layer {layer}.")
        else:
            output_filename = f"combined_spikes_{layer}.csv"
            # Read and merge all files
            # If your files are space-separated (NEST default), use sep='\s+'
            df_list = [pd.read_csv(f, sep=r'\s+', skiprows=3, names=['sender', 'time_ms']) 
                    for f in spike_files]
            combined_df = pd.concat(df_list, ignore_index=True)

            # Save the merged file
            combined_df.sort_values(by='time_ms', inplace=True) # Optional: sort by time
            combined_df.to_csv(os.path.join(data_path, output_filename), index=False)
            print(f"Merged {len(spike_files)} files into {output_filename}")

                    # Delete the original process files
            for f in spike_files:
                os.remove(f)
            print("Original process files deleted.")

def spatial_3d_plot(locations, layers_name):
    """
    Plot the spatial locations of neurons in the cortical column.

    Parameters:
    locations (dict): A dictionary with layer names as keys and lists of spatial locations as values.
    layers_name (list): List containing the names of each layer.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']  # Colors for different layers
    for i, layer in enumerate(layers_name):
        locs = np.array(locations[layer])
        ax.scatter(locs[:, 0]/1000, locs[:, 1]/1000, locs[:, 2]/1000, c=colors[i], label=layer, alpha=0.7, s = 7)

    ax.set_zlim(2.3, 0)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Depth (mm)')
    ax.set_title('Spatial Locations of Neurons in Cortical Column')
    ax.legend(markerscale=5)
    plt.savefig('figures/spatial3D', dpi=300)

def matrix_connectivity_nsyn(layers_name, cortex, annot=False):
    nsyn_array = np.zeros((len(layers_name), len(layers_name)), dtype=int)
    for i, src_layer in enumerate(layers_name):
        for j, tgt_layer in enumerate(layers_name):
            src_nodes = cortex[src_layer]
            tgt_nodes = cortex[tgt_layer]
            conn = nest.GetConnections(src_nodes, tgt_nodes)
            nsyn_array[j, i] = len(conn)
    plt.figure()
    ax = sns.heatmap(nsyn_array, linewidth=0.5, yticklabels = layers_name, xticklabels = layers_name, cmap = "Spectral", annot = annot, cbar_kws={'label': 'Number of Connections'})
    ax.set(xlabel='Source Group', ylabel='Target Group')
    plt.savefig('figures/connectivity.png', dpi=300)

def raster_plot(spikes, sim_time, layer_order, layer_colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Lists to store the tick positions and labels
    yticks_pos = []
    yticks_labels = []
    for layer in layer_order:
        if layer in spikes:
            df = spikes[layer]
            
            unique_neurons = df['sender'].unique()
            n_sample = max(1, int(len(unique_neurons) * 0.1))
            sampled_gids = np.random.choice(unique_neurons, size=n_sample, replace=False)
            df_sampled = df[df['sender'].isin(sampled_gids)]

            layer_center = (np.max(sampled_gids) + np.min(sampled_gids)) / 2
            yticks_pos.append(layer_center)
            yticks_labels.append(layer)
            
            ax.scatter(df_sampled['time_ms'], df_sampled['sender'], s=5, label=layer, color=layer_colors[layer])


    ax.set_yticks(yticks_pos, yticks_labels, fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=15)
    ax.set_ylabel('Neuron Group', fontsize=15)
    ax.set_xlim(sim_time-500, sim_time) # Focus on the time range of interest
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/m1_raster_plot.png', dpi=300, bbox_inches='tight')

def plot_rates(spike_recorders, lname, sim_time, dt=0.1, window_ms=0.1):
    """
    Creates a stacked firing rate plot matching the provided image.
    """
    num_layers = len(spike_recorders)
    # 1. Create subplots with sharex=True and no vertical space
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace=0.0) # This removes the gap between subplots

    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    time_bins = np.arange(0, sim_time, dt)
    start_idx = int(sim_time-500.0 / dt)  # Skip initial transient

    for i, sr in enumerate(spike_recorders):
        ax = axes[i]
        
        # --- Data Processing ---
        data = spike_recorders[lname[i]]
        times = data["time_ms"]
        # Get count of neurons in the specific NodeCollection recorded
        num_neurons = len(np.unique(data.get("sender"))) 
        
        # Calculate rate
        counts, _ = np.histogram(times, bins=time_bins)
        raw_rate = (counts / num_neurons) * (1000.0 / dt)
        
        # Smoothing (adjust sigma to match the "spikiness" of your image)
        sigma = window_ms / dt
        smoothed_rate = gaussian_filter1d(raw_rate, sigma=sigma)
        
        # --- Plotting ---
        ax.plot(time_bins[start_idx:], smoothed_rate[start_idx:], 
                color=colours[i % len(colours)], linewidth=1.5)
        
        # Baseline dashed line (Mean)
        mean_val = np.mean(smoothed_rate[start_idx:])
        ax.axhline(y=mean_val, color='black', linestyle='--', linewidth=1)

        # --- Styling to match the image ---
        ax.annotate(lname[i], xy=(0.92, 0.7), xycoords='axes fraction', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only show the bottom axis for the last plot
        if i < num_layers - 1:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none') 
        else:
            ax.set_xlabel('Time (ms)', fontsize=15)
        
        # Frequency label in the middle of the stack
        if i == num_layers // 2:
            ax.set_ylabel('Frequency (Hz)', fontsize=15)

    plt.tight_layout()
    plt.savefig(f'figures/firing_rates.png', dpi=300)