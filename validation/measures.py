"""
Validation Measures
===================
Pure statistical functions for analysing NEST spike-train outputs.

All functions are decoupled from simulation parameters and accept
pre-loaded NumPy arrays directly, making them easy to unit-test and
reuse across models.

Functions
---------
calculate_firing_rate   Per-neuron mean firing rate.
calculate_isi           Per-neuron inter-spike interval arrays.
calculate_cv            Coefficient of variation of ISIs (Destexhe 2009).
calculate_cv2           Local CV2 of ISIs (Holt et al. 1996).
calculate_kuramoto_order  Time-resolved Kuramoto order parameter R(t).
calculate_sttc          Spike Time Tiling Coefficient (Cutts & Eglen 2014).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Firing rate
# ---------------------------------------------------------------------------

def calculate_firing_rate(senders, spike_times, time_window_ms):
    """
    Compute per-neuron mean firing rates.

    The caller is responsible for restricting ``spike_times`` to the
    desired analysis window before calling this function; ``time_window_ms``
    is used only for rate normalisation.

    Parameters
    ----------
    senders : array-like of int
        Neuron IDs corresponding to each entry in ``spike_times``.
    spike_times : array-like of float
        Spike timestamps (ms).
    time_window_ms : float
        Duration of the analysis window (ms).

    Returns
    -------
    np.ndarray
        Firing rate (Hz) for each unique sender.
    """
    senders = np.asarray(senders)
    spike_times = np.asarray(spike_times)
    firing_rates = []
    for sender in np.unique(senders):
        n_spikes = np.sum(senders == sender)
        firing_rates.append(n_spikes / (time_window_ms / 1000.0))
    return np.array(firing_rates)


# ---------------------------------------------------------------------------
# ISI-based irregularity
# ---------------------------------------------------------------------------

def calculate_isi(senders, spike_times):
    """
    Compute per-neuron inter-spike interval (ISI) sequences.

    Parameters
    ----------
    senders : array-like of int
    spike_times : array-like of float
        Spike timestamps (ms).

    Returns
    -------
    list of np.ndarray
        One array of ISIs per unique sender, in the order returned by
        ``np.unique(senders)``.
    """
    senders = np.asarray(senders)
    spike_times = np.asarray(spike_times)
    isi = []
    for sender in np.unique(senders):
        sender_times = spike_times[senders == sender]
        isi.append(np.diff(sender_times))
    return isi


def calculate_cv(senders, spike_times):
    """
    Coefficient of variation (CV) of ISIs per neuron.

    Defined as ``std(ISI) / mean(ISI)`` following Destexhe (2009).
    Neurons with fewer than two inter-spike intervals receive ``NaN``.

    Parameters
    ----------
    senders : array-like of int
    spike_times : array-like of float

    Returns
    -------
    np.ndarray
        CV for each unique sender (``NaN`` when undefined).
    """
    isi = calculate_isi(senders, spike_times)
    cv_values = []
    for intervals in isi:
        if len(intervals) > 1:
            mean_isi = np.mean(intervals)
            cv_values.append(
                np.std(intervals) / mean_isi if mean_isi != 0 else 0.0
            )
        else:
            cv_values.append(np.nan)
    return np.array(cv_values)


def calculate_cv2(senders, spike_times):
    """
    Local coefficient of variation (CV2) of ISIs per neuron.

    Defined following Holt et al. (1996) as the mean of
    ``2 |ISI_{n+1} - ISI_n| / (ISI_{n+1} + ISI_n)`` across consecutive
    interval pairs.  Neurons with fewer than three ISIs receive ``NaN``.

    Parameters
    ----------
    senders : array-like of int
    spike_times : array-like of float

    Returns
    -------
    np.ndarray
        CV2 for each unique sender (``NaN`` when undefined).
    """
    isi = calculate_isi(senders, spike_times)
    cv2_values = []
    for intervals in isi:
        if len(intervals) > 2:
            i_curr = intervals[:-1]
            i_next = intervals[1:]
            cv2_values.append(
                np.mean(2 * np.abs(i_next - i_curr) / (i_next + i_curr))
            )
        else:
            cv2_values.append(np.nan)
    return np.array(cv2_values)


# ---------------------------------------------------------------------------
# Synchrony
# ---------------------------------------------------------------------------

def calculate_kuramoto_order(senders, spike_times, t_ini, t_fin, dt=0.1):
    """
    Time-resolved Kuramoto order parameter R(t).

    Implements Equations 17–19 from Borges et al. (2019): the phase of
    each neuron is linearly interpolated between successive spikes, and
    ``R(t) = |(1/N) Σ exp(i·Ψ_j(t))|``.

    Parameters
    ----------
    senders : array-like of int
    spike_times : array-like of float
        Spike timestamps (ms).
    t_ini : float
        Start of the analysis window (ms).
    t_fin : float
        End of the analysis window (ms).
    dt : float, optional
        Time resolution of the phase integral (ms).  Default ``0.1``.

    Returns
    -------
    np.ndarray
        ``R(t)`` at each time step in ``[t_ini, t_fin)``.
    """
    senders = np.asarray(senders)
    spike_times = np.asarray(spike_times)
    unique_neurons = np.unique(senders)
    N = len(unique_neurons)

    if N == 0:
        return np.zeros(0)

    times = np.arange(t_ini, t_fin, dt)
    complex_phases_sum = np.zeros(len(times), dtype=complex)

    for neuron_id in unique_neurons:
        spks = spike_times[senders == neuron_id]
        if len(spks) < 2:
            continue
        # Linear interpolation of phase between consecutive spikes (Eq. 18)
        phases = 2 * np.pi * np.interp(times, spks, np.arange(len(spks)))
        complex_phases_sum += np.exp(1j * phases)

    return np.abs(complex_phases_sum / N)


# ---------------------------------------------------------------------------
# Spike Time Tiling Coefficient (STTC)  — Cutts & Eglen (2014)
# ---------------------------------------------------------------------------

def _calculate_t(spikes, n, dt, start, end):
    """
    Total recording time covered by ±``dt`` windows around ``n`` spikes.

    Overlapping windows are merged; windows are clipped to ``[start, end]``.

    Parameters
    ----------
    spikes : np.ndarray
        Sorted spike times (ms).
    n : int
        Number of spikes.
    dt : float
        Half-width of the coincidence window (ms).
    start, end : float
        Recording boundaries (ms).

    Returns
    -------
    float
        Total covered time (ms).
    """
    spikes = np.asarray(spikes)
    time_a = 2 * n * dt

    if n == 1:
        if (spikes[0] - start) < dt:
            time_a -= start - (spikes[0] - dt)
        if (spikes[0] + dt) > end:
            time_a -= spikes[0] + dt - end
    else:
        diffs = np.diff(spikes)
        overlap_mask = diffs < (2 * dt)
        time_a -= np.sum(2 * dt - diffs[overlap_mask])
        if (spikes[0] - start) < dt:
            time_a -= start - (spikes[0] - dt)
        if (end - spikes[-1]) < dt:
            time_a -= spikes[-1] + dt - end

    return time_a


def _calculate_p(spikes1, spikes2, dt):
    """
    Fraction of spikes in ``spikes1`` within ``dt`` of any spike in ``spikes2``.

    Parameters
    ----------
    spikes1, spikes2 : np.ndarray
        Spike time arrays (ms).
    dt : float
        Coincidence window half-width (ms).

    Returns
    -------
    float
    """
    spikes1 = np.asarray(spikes1)
    spikes2 = np.asarray(spikes2)
    if len(spikes1) == 0:
        return 0.0
    count = sum(np.any(np.abs(spikes2 - s) < dt) for s in spikes1)
    return count / len(spikes1)


def calculate_sttc(senders, spike_times, t_ini, t_fin, delta_t, trials=100):
    """
    Estimate the Spike Time Tiling Coefficient (STTC) for a population.

    STTC is computed for ``trials`` randomly drawn neuron pairs following
    Cutts & Eglen (2014).

    Parameters
    ----------
    senders : array-like of int
    spike_times : array-like of float
        Spike timestamps (ms).
    t_ini : float
        Start of the analysis window (ms).
    t_fin : float
        End of the analysis window (ms).
    delta_t : float
        Coincidence window half-width (ms).
    trials : int, optional
        Number of random neuron pairs to sample.  Default ``100``.

    Returns
    -------
    np.ndarray
        STTC value for each sampled pair.  Returns ``[NaN]`` if fewer
        than two neurons are present in the population.
    """
    senders = np.asarray(senders)
    spike_times = np.asarray(spike_times)
    unique_neurons = np.unique(senders)

    if len(unique_neurons) < 2:
        return np.array([np.nan])

    T = t_fin - t_ini
    sttc_values = []
    rng = np.random.default_rng()

    for _ in range(trials):
        n1_id, n2_id = rng.choice(unique_neurons, size=2, replace=False)
        mask1 = (senders == n1_id) & (spike_times >= t_ini) & (spike_times <= t_fin)
        mask2 = (senders == n2_id) & (spike_times >= t_ini) & (spike_times <= t_fin)
        sp1 = spike_times[mask1]
        sp2 = spike_times[mask2]

        if len(sp1) == 0 or len(sp2) == 0:
            sttc_values.append(np.nan)
            continue

        T1 = _calculate_t(sp1, len(sp1), delta_t, t_ini, t_fin) / T
        T2 = _calculate_t(sp2, len(sp2), delta_t, t_ini, t_fin) / T
        P1 = _calculate_p(sp1, sp2, delta_t)
        P2 = _calculate_p(sp2, sp1, delta_t)

        term1 = (P1 - T2) / (1 - T2 * P1) if (1 - T2 * P1) != 0 else 0.0
        term2 = (P2 - T1) / (1 - T1 * P2) if (1 - T1 * P2) != 0 else 0.0
        sttc_values.append(0.5 * (term1 + term2))

    return np.array(sttc_values)
