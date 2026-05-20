import parameters as params
import numpy as np
import scipy.ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def calculate_firing_rate(senders, spike_times, time_window):
    """
    Calculate the firing rate of a neuron population.

    Parameters:
    senders (list or np.array): A list or array of sender IDs.
    spike_times (list or np.array): A list or array of spike times.
    time_window (float): The duration of the time window (in ms) over which to calculate the firing rate.

    Returns:
    np.array: An array of firing rates for each neuron.
    """
    senders = np.array(senders)
    spike_times = np.array(spike_times)
    spike_times = spike_times[spike_times > params.simulation_time - time_window]  # Consider only spikes in the last time_window ms
    firing_rates = []
    
    for sender in np.unique(senders):
        sender_spike_times = spike_times[senders == sender]
        firing_rate = len(sender_spike_times) / (time_window / 1000.0)  # Convert ms to seconds
        firing_rates.append(firing_rate)
    
    return np.array(firing_rates)

def calculate_isi(senders, spike_times):
    """
    Calculate the inter-spike intervals (ISI) of a neuron population.

    Parameters:
    senders (list or np.array): A list or array of sender IDs.
    spike_times (list or np.array): A list or array of spike times.

    Returns:
    np.array: An array of inter-spike intervals.
    """
    senders = np.array(senders)
    spike_times = np.array(spike_times)
    isi = []
    for sender in np.unique(senders):
        sender_spike_times = spike_times[senders == sender]
        isi.append(np.diff(sender_spike_times))
    return isi

def calculate_cv(senders, spike_times):
    """
    Calculate the coefficient of variation (CV) of inter-spike intervals (ISI) for a neuron population, according to Destexhe (2009).

    Parameters:
    senders (list or np.array): A list or array of sender IDs.
    spike_times (list or np.array): A list or array of spike times.
    N_neurons (int): The number of neurons in the population.

    Returns:
    cv_values (list or np.array): A list or array with the CV values for each sender.
    """
    isi = calculate_isi(senders, spike_times)
    cv_values = []
    for intervals in isi:
        if len(intervals) > 1:  # Ensure there are enough intervals to calculate CV
            mean_isi = np.mean(intervals)
            std_isi = np.std(intervals)
            cv_values.append(std_isi / mean_isi if mean_isi != 0 else 0)
        else:
            cv_values.append(np.nan)  # If there is only one interval, CV is not defined
    return np.array(cv_values)

def calculate_cv2(senders, spike_times):
    """
    Calculate the CV2 of inter-spike intervals (ISI) for a neuron population, according to Holt et al. (1996).

    Parameters:
    senders (list or np.array): A list or array of sender IDs.
    spike_times (list or np.array): A list or array of spike times.

    Returns:
    cv2_values (list or np.array): A list or array with the CV2 values for each sender.
    """
    isi = calculate_isi(senders, spike_times)
    cv2_values = []
    
    for intervals in isi:
        if len(intervals) > 2: # Ensure there are enough intervals to calculate CV2
            i_curr = intervals[:-1]
            i_next = intervals[1:]
            mean_cv2 = np.mean(2 * np.abs(i_next - i_curr) / (i_next + i_curr))
            cv2_values.append(mean_cv2)
        else:
            cv2_values.append(np.nan)  # If there are not enough intervals, CV2 is not defined
            
    return np.array(cv2_values)

def calculate_kuramoto_order(senders, spike_times, t_ini, t_fin, dt=0.1):
    """
    Calculates the time-averaged Kuramoto Order Parameter R based on equations 17-19 from Borges (2019).
    
    Parameters:
    senders (np.array): Array of neuron IDs.
    spike_times (np.array): Array of spike timestamps (ms).
    t_ini (float): Start time for analysis.
    t_fin (float): End time for analysis.
    dt (float): Time resolution for the integral (default 0.1 ms).
    
    Returns:
    float: The time-averaged synchrony R.
    """
    unique_neurons = np.unique(senders)
    N = len(unique_neurons)
    
    if N == 0:
        return 0.0

    # Define the time vector for the integral (Eq. 19)
    times = np.arange(t_ini, t_fin, dt)
    complex_phases_sum = np.zeros(len(times), dtype=complex)

    for neuron_id in unique_neurons:
        # Get spikes for this specific neuron within the range
        spks = spike_times[senders == neuron_id]
        
        # We need at least two spikes to define an interval for the phase (Eq. 18)
        if len(spks) < 2:
            continue
            
        # Eq. 18: Linear interpolation of phase between spikes
        # np.interp handles the "t - t_j,m / t_j,m+1 - t_j,m" logic automatically
        # by mapping the time range to the index range [0, 1, 2... len(spks)-1]
        phases = 2 * np.pi * np.interp(times, spks, np.arange(len(spks)))
        
        # Eq. 17 (inner part): exp(i * Psi)
        complex_phases_sum += np.exp(1j * phases)

    # Eq. 17: rho(t) = | (1/N) * sum(exp(i * Psi)) |
    rho_t = np.abs(complex_phases_sum / N)

    # Eq. 19: Time average R = (1 / Delta_T) * integral(rho(t) dt)
    # We use np.mean as a discrete approximation of the normalized integral
    #R = np.mean(rho_t)
    
    return rho_t

def calculate_t(spikes, n, dt, start, end):
    """
    Calculate the total time covered by the spikes, accounting for overlaps and edges.

    Parameters:
    spikes (np.array): Array of spike times.
    n (int): Number of spikes.
    dt (float): Time window for each spike (ms).
    start (float): Start time of the recording (ms).
    end (float): End time of the recording (ms).

    Returns:
    float: Total time covered by the spikes (ms).

    """
    spikes = np.array(spikes)

    # Maximum theoretical time covered (2*dt per spike)
    time_a = 2 * n * dt

    if n == 1:
        if (spikes[0] - start) < dt:
            time_a -= (start - (spikes[0] - dt))
        if (spikes[0] + dt) > end:
            time_a -= (spikes[0] + dt - end)
    else:
        # Subtract overlaps between consecutive spikes
        diffs = np.diff(spikes)
        overlap_mask = diffs < (2 * dt)
        time_a -= np.sum(2 * dt - diffs[overlap_mask])

        # Correct for edges (start and end of recording)
        if (spikes[0] - start) < dt:
            time_a -= (start - (spikes[0] - dt))
        if (end - spikes[-1]) < dt:
            time_a -= (spikes[-1] + dt - end)

    return time_a

def calculate_p(spikes1, spikes2, dt):
    """
    Calculate the probability of a spike occurring within a time window.

    Parameters:
    spikes1 (np.array): Array of spike times for neuron 1.
    spikes2 (np.array): Array of spike times for neuron 2.
    n1 (int): Number of spikes for neuron 1.
    n2 (int): Number of spikes for neuron 2.
    dt (float): Time window for each spike (ms).

    Returns:
    float: Probability of a spike occurring within the time window.
    """

    spikes1 = np.array(spikes1)
    spikes2 = np.array(spikes2)

    count = 0
    for spike in spikes1:
        if np.any(np.abs(spikes2 - spike) < dt):
            count += 1
    return count / len(spikes1) if len(spikes1) > 0 else 0

def calculate_sttc(senders, spike_times, t_ini, t_fin, delta_t, trials=100):
    """
    Calculate the Spike Time Tiling Coefficient (STTC) for a pair of neurons, according to Cutts and Eglen (2014).

    Parameters:
    senders (np.array): Array of neuron IDs.
    spike_times (np.array): Array of spike timestamps (ms).
    t_ini (float): Start time for analysis.
    t_fin (float): End time for analysis.
    delta_t (float): Time window for considering spikes as coincident (ms).

    Returns:
    float: The STTC value for the pair of neurons.
    """
    unique_neurons = np.unique(senders)
    
    if len(unique_neurons) < 2:
        return np.nan  # Not enough neurons to calculate STTC

    sttc_values = []
    for _ in range(trials):
        # Get spike times for the first two neurons
        neuron1_spikes = spike_times[senders == np.random.choice(unique_neurons)]
        neuron2_spikes = spike_times[senders == np.random.choice(unique_neurons)]

        # Filter spikes within the time range
        neuron1_spikes = neuron1_spikes[(neuron1_spikes >= t_ini) & (neuron1_spikes <= t_fin)]
        neuron2_spikes = neuron2_spikes[(neuron2_spikes >= t_ini) & (neuron2_spikes <= t_fin)]

        # Calculate the number of spikes for each neuron
        N1 = len(neuron1_spikes)
        N2 = len(neuron2_spikes)

        if N1 == 0 or N2 == 0:
            return np.nan  # Cannot calculate STTC if one neuron has no spikes

        T = t_fin - t_ini

        # Calculate the proportion of total recording time covered by spikes for each neuron
        T1 = calculate_t(neuron1_spikes, N1, delta_t, t_ini, t_fin) / T
        T2 = calculate_t(neuron2_spikes, N2, delta_t, t_ini, t_fin) / T

        # Calculate the proportion of spikes from each neuron that are within delta_t of a spike from the other neuron
        P1 = calculate_p(neuron1_spikes, neuron2_spikes, delta_t)
        P2 = calculate_p(neuron2_spikes, neuron1_spikes, delta_t)

        term1 = (P1 - T2) / (1 - T2 * P1) if (1 - T2 * P1) != 0 else 0
        term2 = (P2 - T1) / (1 - T1 * P2) if (1 - T1 * P2) != 0 else 0

        STTC = 0.5 * (term1 + term2)
        sttc_values.append(STTC)
    
    return np.array(sttc_values)
    