import nest
import parameters as params
import numpy as np


def spatial_location(n_layer, layers_name, num_cols, X_range, Y_range, Z_range, spacing):
    """
    Calculate the spatial location of each neuron in the cortical column.

    Parameters:
    n_layer (list): List containing the number of neurons in each layer.
    layers_name (list): List containing the names of each layer.
    num_cols (int): Number of columns in the cortical model.

    Returns:
    dict: A dictionary with layer names as keys and lists of spatial locations as values.
    """
    spatial_locations = {}
    
    # Define the z-boundaries for each layer based on the number of neurons and the total Z range
    z_boundaries = [0]  # Initialize with the starting boundary (0)
    for i in range(0, len(layers_name), 2):
        n = n_layer[i] + n_layer[i+1] 
        z_boundaries.append(z_boundaries[-1] + n/sum(n_layer) * Z_range)  # Calculate the next boundary based on the proportion of neurons in the layer

    # Generate spatial locations for each neuron in each layer and column
    dim = np.sqrt(num_cols)  # Assuming a square arrangement of columns
    for i in range(len(layers_name)):
        layer_name = layers_name[i]
        num_neurons = n_layer[i]
        for col in range(num_cols):
            
            layer_pos = i // 2  # Determine the layer position (0 for layers 2/3, 1 for layer 4, etc.)
            # Calculate the z-coordinate for the current layer
            z_min = z_boundaries[layer_pos]
            z_max = z_boundaries[layer_pos + 1]
            
            # Generate spatial locations for neurons in the current layer
            locations = []
            for _ in range(num_neurons):
                x = np.random.randint(X_range) + (col % dim) * (X_range + spacing)  # Random x-coordinate within the specified range
                y = np.random.randint(Y_range) + np.floor(col / dim) * (Y_range + spacing)  # Random y-coordinate within the specified range
                z = np.random.randint(z_min, z_max)  # Random z-coordinate within the layer boundaries
                locations.append((x, y, z))
            
            spatial_locations[layer_name] = spatial_locations.get(layer_name, []) + locations

    return spatial_locations

def build_nest_synapses(nodes_dict, conn_table, connection_type='random'):
    """
    nodes_dict: Dictionary of NEST NodeCollections (e.g., {'2/3E': NC, '2/3I': NC...})
    conn_table: The pandas DataFrame provided
    connection_type: Type of connection to build ('random' or 'local')
    """
    
    for _, r in conn_table.iterrows():
        # Identify Source and Target strings to match your dictionary keys
        src_key = f"{r['Source']}{r['SourceType']}"
        tgt_key = f"{r['Target']}{r['TargetType']}"
        
        src_nodes = nodes_dict[src_key]
        tgt_nodes = nodes_dict[tgt_key]

        # Calculate nsyn using Peter's Rule
        # nsyn = int(ln(1-Pmax) / ln(1 - 1/(Nsrc*Ntgt)))
        n_src = len(src_nodes)
        n_tgt = len(tgt_nodes)
        
        p_max = r['Pmax']
        # Avoid calculating nsyn if p_max is zero
        if p_max == 0: 
            continue 
        
        nsyn = int(np.log(1.0 - p_max) / np.log(1.0 - (1.0 / (n_src * n_tgt))))

        # Clean numeric values from strings (e.g., '87.8*pA' -> 87.8)
        def clean_val(val):
            if isinstance(val, str):
                return float(val.split('*')[0])
            return val

        weight_mean = clean_val(r['Weight'])
        weight_std = clean_val(r['Wstd'])
        delay_mean = r['Delay']
        delay_std = r['Dstd']
        radius = r['Radius'] * 1e6

        if connection_type == 'random':
            conn_spec = {'rule': 'fixed_total_number', 
                        'N': nsyn,
                        'allow_multapses': True,
                        'allow_autapses': False}
            syn_spec = {'synapse_model': 'static_synapse','weight': nest.random.normal(mean=weight_mean, std=weight_std), 'delay': nest.math.redraw(nest.random.normal(mean=delay_mean, std=delay_std), 0.1, 100.0),
                        'receptor_type': 0}
            nest.Connect(src_nodes, tgt_nodes, conn_spec, syn_spec)
            print(f"Connected {src_key} to {tgt_key} with {nsyn} synapses (Random)")
            
        elif connection_type == 'local':
            # do something here
            pass

def background_noise(cortex, bg_freq, bg_layer):
    """
    Connect background noise to each layer of the cortex.

    Parameters:
    cortex (dict): A dictionary containing the neuron groups for each layer.
    bg_freq (float): The base frequency of the background noise.
    bg_layer (list): A list of scaling factors for the background noise for each layer.
    """
    for i, layer in enumerate(params.layers_name):
        bg = nest.Create('poisson_generator', params={'rate': bg_freq * bg_layer[i]})
        nest.Connect(bg, cortex[layer], syn_spec={'weight': params.w_ex, 'delay': params.d_ex})

def connect_spike_detectors(cortex):
    """
    Connect spike detectors to each layer of the cortex.

    Parameters:
    cortex (dict): A dictionary containing the neuron groups for each layer.
    """
    for layer in params.layers_name:
        # Create a Spike Recorder
        if layer == '2/3E':
            spike_det = nest.Create("spike_recorder", params={"record_to": "ascii", "label": f"spikes_23E"})
        elif layer == '2/3I':
            spike_det = nest.Create("spike_recorder", params={"record_to": "ascii", "label": f"spikes_23I"})
        else:
            spike_det = nest.Create("spike_recorder", params={"record_to": "ascii", "label": f"spikes_{layer}"})
        nest.Connect(cortex[layer], spike_det)