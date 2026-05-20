import matplotlib.pyplot as plt
import nest
import glob
import os
import seaborn as sns
import numpy as np
import pandas as pd
import parameters as params
import scipy
from scipy.ndimage import gaussian_filter1d
import measures as ms

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

def plot_rates(spike_recorders, layer_names, sim_time, dt=0.1, window_ms=0.1):
    num_layers = len(spike_recorders)
    # Create subplots with sharex=True and no vertical space
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace=0.0) # This removes the gap between subplots

    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    time_bins = np.arange(0, sim_time, dt)
    start_idx = int(sim_time-500.0 / dt)  # Skip initial transient

    for i, sr in enumerate(spike_recorders):
        ax = axes[i]
        
        data = spike_recorders[layer_names[i]]
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

        # --- Styling to match the image ---
        ax.annotate(layer_names[i], xy=(0.92, 0.7), xycoords='axes fraction', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only show the bottom axis for the last plot
        if i < num_layers - 1:
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_ticks_position('none') 
        else:
            ax.set_xlabel('Time (ms)', fontsize=15)
        
        # Frequency label in the middle of the stack
        if i == num_layers // 2:
            ax.set_ylabel('Frequency (Hz)', fontsize=15)

    plt.tight_layout()
    plt.savefig(f'figures/firing_rates.png', dpi=300)

def CV_boxplot(spike_recorders, layer_names, method='CV'):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']

    CVs = []
    for layer in layer_names:
        data = spike_recorders[layer]
        times = data["time_ms"]
        senders = data["sender"]
        
        if method == 'CV':
            # Calculate CVs for this layer
            cv_values = ms.calculate_cv(senders, times)
        elif method == 'CV2':
            cv_values = ms.calculate_cv2(senders, times)
        
        CVs.append(cv_values)
  
    # Clean NaNs (neurons with < 3 spikes)
    cleaned_CVs = []
    for group in CVs:
        cleaned_CVs.append([x for x in group if not np.isnan(x)])
    
    # Reverse to match layout
    local_cvs = cleaned_CVs[::-1]
    local_names = list(layer_names)[::-1]
    local_cols = colours[::-1]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(local_cvs, vert=False, patch_artist=True)
    
    ax.set_yticklabels(local_names)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Irregularity (CV)')
    ax.set_ylabel('Neuron Groups')
    ax.set_title(f'CV of Inter-Spike Intervals ({method})')
    
    for patch, color in zip(bp['boxes'], local_cols):
        patch.set_facecolor(color)
    
    for median in bp['medians']:
        median.set(color='black')
        
    plt.savefig(f'figures/CVs_box_{method}.png', dpi=300, bbox_inches='tight')

def FR_boxplot(spike_recorders, layer_names):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    FRs = []
    for layer in layer_names:
        data = spike_recorders[layer]
        times = data["time_ms"]
        senders = data["sender"]
        firing_rates = ms.calculate_firing_rate(senders, times, 600.0)  # Calculate firing rate over the last 500 ms
        FRs.append(firing_rates)

    # Reverse to match layout
    FRs = FRs[::-1]
    layer_names = list(layer_names)[::-1]
    colours = colours[::-1]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(FRs, vert=False, patch_artist=True)
    
    ax.set_yticklabels(layer_names)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Neuron Groups')
    ax.set_title('Firing Rates of Neuron Groups')
    
    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    
    for median in bp['medians']:
        median.set(color='black')
        
    plt.savefig('figures/firing_rates_box.png', dpi=300, bbox_inches='tight')

def kuramoto_order_boxplot(spike_recorders, layer_names):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    order_params = []
    for layer in layer_names:
        data = spike_recorders[layer]
        times = data["time_ms"]
        senders = data["sender"]
        order_param = ms.calculate_kuramoto_order(senders, times, params.simulation_time - 500.0, params.simulation_time)  # Calculate over the last 500 ms
        order_params.append(order_param)

    # Reverse to match layout
    order_params = order_params[::-1]
    layer_names = list(layer_names)[::-1]
    colours = colours[::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(order_params, vert=False, patch_artist=True)
    ax.set_yticklabels(layer_names)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Kuramoto Order Parameter')
    ax.set_ylabel('Neuron Groups')
    ax.set_title('Kuramoto Order Parameter of Neuron Groups')

    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='black')
    plt.savefig('figures/kuramoto_order_box.png', dpi=300, bbox_inches='tight')

def sttc_boxplot(spike_recorders, layer_names, delta_t=5.0):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    sttc_values = []
    for layer in layer_names:
        data = spike_recorders[layer]
        times = data["time_ms"]
        senders = data["sender"]
        sttc = ms.calculate_sttc(senders, times, params.simulation_time - 500.0, params.simulation_time, delta_t)
        sttc_values.append(sttc)

    # Reverse to match layout
    sttc_values = sttc_values[::-1]
    layer_names = list(layer_names)[::-1]
    colours = colours[::-1]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(sttc_values, vert=False, patch_artist=True)
    ax.set_yticklabels(layer_names)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('STTC')
    ax.set_ylabel('Neuron Groups')
    ax.set_title(f'Spike Time Tiling Coefficient (delta_t={delta_t} ms)')

    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='black')
    plt.savefig('figures/sttc_box.png', dpi=300, bbox_inches='tight')

def plot_psd(spike_recorders, layer_names, sim_time, dt=0.1, window_ms=0.1):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    time_bins = np.arange(0, sim_time, dt)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sr in enumerate(spike_recorders):  
        data = spike_recorders[layer_names[i]]
        times = data["time_ms"]
        # Get count of neurons in the specific NodeCollection recorded
        num_neurons = len(np.unique(data.get("sender"))) 
        
        # Calculate rate
        counts, _ = np.histogram(times, bins=time_bins)
        raw_rate = (counts / num_neurons) * (1000.0 / dt)
        
        # Smoothing (adjust sigma to match the "spikiness" of your image)
        sigma = 1.5
        smoothed_rate = gaussian_filter1d(raw_rate, sigma=sigma)
        smoothed_rate = raw_rate
        
        sample_rate = (1000.0 / dt)
        fourier_transform = np.fft.rfft(smoothed_rate)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        frequency = np.linspace(0, sample_rate/2, len(power_spectrum))
        smooth_powerspectrum = scipy.ndimage.gaussian_filter1d(power_spectrum, sigma)
        ax.plot(frequency[3:], smooth_powerspectrum[3:], color=colours[i % len(colours)], label=layer_names[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Frequency (Hz)', fontsize=15)
    ax.set_ylabel('Power', fontsize=15)
    ax.set_title('Power Spectral Density of Firing Rates', fontsize=15)
    ax.set_xlim(0, 100)  # Focus on frequencies up to 100 Hz
    ax.legend()
    plt.savefig('figures/psd.png', dpi=300, bbox_inches='tight')