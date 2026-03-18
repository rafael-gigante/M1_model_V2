import parameters as params

def calculate_firing_rate(spike_times, time_window):
    """
    Calculate the firing rate of a neuron given its spike times and a specified time window.

    Parameters:
    spike_times (list): A list of spike times (in ms) for a single neuron.
    time_window (float): The duration of the time window (in ms) over which to calculate the firing rate.

    Returns:
    float: The firing rate in Hz (spikes per second).
    """
    if time_window <= 0:
        raise ValueError("Time window must be greater than zero.")
    
    # Count the number of spikes that occur within the time window
    spike_count = sum(1 for t in spike_times if t <= time_window)
    
    # Convert time window from ms to seconds for firing rate calculation
    time_window_sec = time_window / 1000.0
    
    # Calculate firing rate in Hz
    firing_rate = spike_count / time_window_sec
    
    return firing_rate