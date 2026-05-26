"""
Validation Plots
================
Post-simulation visualisation functions for the M1 cortical column network.

Spike-based analysis functions accept a ``spikes`` dictionary whose
values are pandas DataFrames with ``sender`` (int) and ``time_ms``
(float) columns, keyed by layer name (e.g. ``'2/3E'``).

Current-based functions (e.g. :func:`plot_lfp_psd`) accept a
``currents`` dictionary whose values are DataFrames with columns
``sender``, ``time_ms``, ``I_syn_ex``, and ``I_syn_in``, as produced
by :meth:`M1Network.save_current_data`.

Simulation parameters that affect the analysis window (e.g.
``sim_time``) are passed explicitly as arguments so that this module
has no dependency on ``parameters.py``.

Functions that require an active NEST kernel are clearly marked and
should be called before ``nest.ResetKernel``.

Typical usage
-------------
>>> import plots as pl
>>> pl.raster_plot(spikes, sim_time, layer_order)
>>> pl.FR_boxplot(spikes, layer_order)
>>> pl.plot_lfp_psd(currents, layer_order, dt=0.1)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

import measures as ms


# ---------------------------------------------------------------------------
# Canonical layer style
# ---------------------------------------------------------------------------

LAYER_COLORS = {
    "2/3E": "#8ca465", "2/3I": "#dec47c",
    "4E":   "#487a99", "4I":   "#d19f7f",
    "5E":   "#385f49", "5I":   "#ae5b5e",
    "6E":   "#2e4876", "6I":   "#85588c",
}

_COLOR_LIST = list(LAYER_COLORS.values())


# ---------------------------------------------------------------------------
# Spatial plot  (requires post-create_nodes data, no NEST kernel needed)
# ---------------------------------------------------------------------------

def spatial_3d_plot(locations, layers_name):
    """
    3-D scatter plot of neuron spatial positions.

    Parameters
    ----------
    locations : dict[str, list[tuple]]
        ``{layer_name: [(x, y, z), ...]}`` positions in µm, as stored
        in :attr:`M1Network.spatial_locations`.
    layers_name : list[str]
        Ordered list of layer names.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, layer in enumerate(layers_name):
        locs = np.array(locations[layer])
        ax.scatter(
            locs[:, 0] / 1000,
            locs[:, 1] / 1000,
            locs[:, 2] / 1000,
            c=_COLOR_LIST[i],
            label=layer,
            alpha=0.7,
            s=7,
        )

    ax.set_zlim(2.3, 0)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Depth (mm)")
    ax.set_title("Spatial Locations of Neurons in Cortical Column")
    ax.legend(markerscale=5)
    plt.savefig("figures/spatial3D.png", dpi=300)


# ---------------------------------------------------------------------------
# Connectivity matrix  (requires active NEST kernel)
# ---------------------------------------------------------------------------

def matrix_connectivity_nsyn(layers_name, n_layer, num_cols, conn_table, annot=False):
    """
    Heat-map of expected synapse counts derived from Peter's Rule.

    Computes synapse counts analytically — no active NEST kernel required.
    The result matches exactly what :meth:`M1Network.connect` creates:
    for each of the ``num_cols²`` column pairs, Peter's Rule is applied to
    the per-column population sizes and the counts are summed.

    Parameters
    ----------
    layers_name : list[str]
        Ordered layer names (e.g. ``['2/3E', '2/3I', ...]``).
    n_layer : list[int]
        Number of neurons per layer *per column*, in the same order as
        ``layers_name``.
    num_cols : int
        Number of cortical columns.
    conn_table : pd.DataFrame
        Connection table with at least the columns ``Source``,
        ``SourceType``, ``Target``, ``TargetType``, ``Pmax``
        (as loaded by :class:`M1Network`).
    annot : bool, optional
        Annotate each cell with its synapse count.  Default ``False``.
    """
    # Neurons per layer per column, keyed by layer name
    n_per_col = dict(zip(layers_name, n_layer))

    layer_idx = {name: i for i, name in enumerate(layers_name)}
    n = len(layers_name)
    nsyn_array = np.zeros((n, n), dtype=int)

    for _, row in conn_table.iterrows():
        src_key = f"{row['Source']}{row['SourceType']}"
        tgt_key = f"{row['Target']}{row['TargetType']}"
        p_max = float(row["Pmax"])

        if p_max == 0:
            continue
        if src_key not in layer_idx or tgt_key not in layer_idx:
            continue

        n_src = n_per_col[src_key]
        n_tgt = n_per_col[tgt_key]

        # Peter's Rule applied per column pair, then summed over all C² pairs
        # — identical to what network.py connect() produces.
        nsyn_per_pair = int(
            np.log(1.0 - p_max) / np.log(1.0 - 1.0 / (n_src * n_tgt))
        )
        nsyn_array[layer_idx[tgt_key], layer_idx[src_key]] = (
            num_cols ** 2 * nsyn_per_pair
        )

    fig, ax = plt.subplots()
    sns.heatmap(
        nsyn_array,
        linewidth=0.5,
        yticklabels=layers_name,
        xticklabels=layers_name,
        cmap="Spectral",
        annot=annot,
        cbar_kws={"label": "Number of synapses"},
        ax=ax,
    )
    ax.set_xlabel("Source group")
    ax.set_ylabel("Target group")
    plt.tight_layout()
    plt.savefig("figures/connectivity.png", dpi=300)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _layer_boxplot(data_list, layer_order, xlabel, title, save_path):
    """
    Shared horizontal box-plot layout with reversed layer order.

    Parameters
    ----------
    data_list : list of array-like
        One array per layer, in the same order as ``layer_order``.
    layer_order : list[str]
    xlabel, title, save_path : str
    """
    colours = list(reversed(_COLOR_LIST[: len(layer_order)]))
    data_rev = list(reversed(data_list))
    labels_rev = list(reversed(layer_order))

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(data_rev, vert=False, patch_artist=True)

    ax.set_yticklabels(labels_rev)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Neuron groups")
    ax.set_title(title)

    for patch, color in zip(bp["boxes"], colours):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set(color="black")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Post-simulation analysis plots
# ---------------------------------------------------------------------------

def raster_plot(spikes, sim_time, layer_order, layer_colors=None, window_ms=500.0):
    """
    Raster plot of sampled spikes from each layer.

    10 % of neurons are randomly sampled per layer to reduce overplotting.
    Only spikes in the last ``window_ms`` ms of the simulation are shown.

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    sim_time : float
        Total simulation time (ms).
    layer_order : list[str]
    layer_colors : dict[str, str], optional
        Hex colour per layer.  Defaults to :data:`LAYER_COLORS`.
    window_ms : float, optional
        Width of the displayed time window.  Default ``500.0``.
    """
    if layer_colors is None:
        layer_colors = LAYER_COLORS

    fig, ax = plt.subplots(figsize=(8, 8))
    yticks_pos, yticks_labels = [], []

    for layer in layer_order:
        if layer not in spikes:
            continue
        df = spikes[layer]
        df = df[df["time_ms"] > sim_time - window_ms]

        unique_neurons = df["sender"].unique()
        n_sample = max(1, int(len(unique_neurons) * 0.1))
        sampled_gids = np.random.choice(unique_neurons, size=n_sample, replace=False)
        df_sampled = df[df["sender"].isin(sampled_gids)]

        layer_center = (sampled_gids.max() + sampled_gids.min()) / 2
        yticks_pos.append(layer_center)
        yticks_labels.append(layer)

        ax.scatter(
            df_sampled["time_ms"],
            df_sampled["sender"],
            s=5,
            color=layer_colors[layer],
            label=layer,
        )

    ax.set_yticks(yticks_pos, yticks_labels, fontsize=12)
    ax.set_xlabel("Time (ms)", fontsize=15)
    ax.set_ylabel("Neuron group", fontsize=15)
    ax.set_xlim(sim_time - window_ms, sim_time)
    ax.spines[["right", "top"]].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("figures/m1_raster_plot.png", dpi=300, bbox_inches="tight")


def plot_rates(spikes, layer_order, sim_time, dt=0.1, window_ms=500.0, smooth_sigma=1.0):
    """
    Stacked per-layer population firing-rate traces.

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    layer_order : list[str]
    sim_time : float
        Total simulation time (ms).
    dt : float, optional
        Time bin width (ms).  Default ``0.1``.
    window_ms : float, optional
        Display window (ms from end of simulation).  Default ``500.0``.
    smooth_sigma : float, optional
        Gaussian smoothing sigma in bins.  Default ``1.0``.
    """
    n_layers = len(layer_order)
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, 8), sharex=True)
    plt.subplots_adjust(hspace=0.0)

    time_bins = np.arange(0, sim_time, dt)
    start_idx = int((sim_time - window_ms) / dt)

    for i, layer in enumerate(layer_order):
        ax = axes[i]
        data = spikes[layer]
        times = data["time_ms"].values
        num_neurons = data["sender"].nunique()

        counts, _ = np.histogram(times, bins=time_bins)
        rate = gaussian_filter1d(
            (counts / num_neurons) * (1000.0 / dt), sigma=smooth_sigma
        )

        ax.plot(time_bins[start_idx:-1], rate[start_idx:], color=_COLOR_LIST[i], linewidth=1.5)
        ax.annotate(
            layer, xy=(0.92, 0.7), xycoords="axes fraction",
            fontsize=12, fontweight="bold",
        )
        ax.spines[["right", "top"]].set_visible(False)

        if i < n_layers - 1:
            ax.tick_params(axis="x", bottom=False)
        else:
            ax.set_xlabel("Time (ms)", fontsize=15)

        if i == n_layers // 2:
            ax.set_ylabel("Frequency (Hz)", fontsize=15)

    plt.tight_layout()
    plt.savefig("figures/firing_rates.png", dpi=300)


def CV_boxplot(spikes, layer_order, method="CV"):
    """
    Horizontal box-plot of per-neuron ISI irregularity.

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    layer_order : list[str]
    method : {'CV', 'CV2'}
        Irregularity measure to use.
    """
    cv_fn = ms.calculate_cv if method == "CV" else ms.calculate_cv2
    cvs = []
    for layer in layer_order:
        data = spikes[layer]
        vals = cv_fn(data["sender"].values, data["time_ms"].values)
        cvs.append([v for v in vals if not np.isnan(v)])

    _layer_boxplot(
        cvs,
        layer_order,
        xlabel=f"Irregularity ({method})",
        title=f"CV of Inter-Spike Intervals ({method})",
        save_path=f"figures/CVs_box_{method}.png",
    )


def FR_boxplot(spikes, layer_order, window_ms=500.0):
    """
    Horizontal box-plot of per-neuron firing rates.

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    layer_order : list[str]
    window_ms : float, optional
        Analysis window length (ms) used for rate normalisation.
        Default ``500.0``.
    """
    frs = []
    for layer in layer_order:
        data = spikes[layer]
        frs.append(
            ms.calculate_firing_rate(
                data["sender"].values, data["time_ms"].values, window_ms
            )
        )

    _layer_boxplot(
        frs,
        layer_order,
        xlabel="Firing rate (Hz)",
        title="Firing Rates of Neuron Groups",
        save_path="figures/firing_rates_box.png",
    )


def kuramoto_order_boxplot(spikes, layer_order, sim_time, window_ms=500.0):
    """
    Horizontal box-plot of the time-resolved Kuramoto order parameter R(t).

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    layer_order : list[str]
    sim_time : float
        Total simulation time (ms).
    window_ms : float, optional
        Analysis window (ms from end of simulation).  Default ``500.0``.
    """
    t_ini = sim_time - window_ms
    order_params = []
    for layer in layer_order:
        data = spikes[layer]
        rho_t = ms.calculate_kuramoto_order(
            data["sender"].values, data["time_ms"].values, t_ini, sim_time
        )
        order_params.append(rho_t)

    _layer_boxplot(
        order_params,
        layer_order,
        xlabel="Kuramoto order parameter R",
        title="Kuramoto Order Parameter of Neuron Groups",
        save_path="figures/kuramoto_order_box.png",
    )


def sttc_boxplot(spikes, layer_order, sim_time, delta_t=5.0, window_ms=500.0):
    """
    Horizontal box-plot of Spike Time Tiling Coefficients.

    Parameters
    ----------
    spikes : dict[str, pd.DataFrame]
    layer_order : list[str]
    sim_time : float
        Total simulation time (ms).
    delta_t : float, optional
        Coincidence window half-width (ms).  Default ``5.0``.
    window_ms : float, optional
        Analysis window (ms from end of simulation).  Default ``500.0``.
    """
    t_ini = sim_time - window_ms
    sttc_values = []
    for layer in layer_order:
        data = spikes[layer]
        sttc_values.append(
            ms.calculate_sttc(
                data["sender"].values, data["time_ms"].values,
                t_ini, sim_time, delta_t,
            )
        )

    _layer_boxplot(
        sttc_values,
        layer_order,
        xlabel="STTC",
        title=f"Spike Time Tiling Coefficient (δt = {delta_t} ms)",
        save_path="figures/sttc_box.png",
    )


def plot_lfp(
    currents,
    layer_order,
    dt=0.1,
    smooth_sigma=0.0,
    window_ms=None,
):
    """
    Stacked time-series plot of the LFP proxy for each layer.

    The LFP proxy is the sum of absolute synaptic currents over all neurons
    at each recorded time step:

    .. math::

        \\text{LFP}(t) = \\sum_i \\left( |I_{\\text{ex},i}(t)| + |I_{\\text{in},i}(t)| \\right)

    Each layer is shown in its own panel (shared x-axis) so that
    per-layer amplitude differences are visible without overlap.

    Parameters
    ----------
    currents : dict[str, pd.DataFrame]
        Per-layer DataFrames with columns ``sender``, ``time_ms``,
        ``I_syn_ex``, ``I_syn_in``, as produced by
        :meth:`M1Network.save_current_data`.
    layer_order : list[str]
        Ordered list of layer names to plot.
    dt : float, optional
        Multimeter recording interval (ms).  Must match the ``interval``
        argument passed to :meth:`M1Network.create_current_recorders`.
        Default ``0.1``.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in samples) applied to the LFP signal
        before plotting.  ``0`` disables smoothing.  Default ``0``.
    window_ms : float or None, optional
        If given, only the last ``window_ms`` ms of the recording are
        displayed.  ``None`` (default) shows the full recording.
    """
    n_layers = len(layer_order)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.0)

    for i, layer in enumerate(layer_order):
        ax = axes[i]
        data = currents[layer]

        # LFP proxy: Σ |I_syn_ex| + Σ |I_syn_in| per time step
        lfp = (
            data
            .assign(_abs=np.abs(data["I_syn_ex"]) + np.abs(data["I_syn_in"]))
            .groupby("time_ms")["_abs"]
            .sum()
            .sort_index()
        )
        t = lfp.index.values
        sig = lfp.values

        if smooth_sigma > 0:
            sig = scipy.ndimage.gaussian_filter1d(sig, smooth_sigma)

        if window_ms is not None:
            mask = t >= t[-1] - window_ms
            t, sig = t[mask], sig[mask]

        ax.plot(t, sig, color=_COLOR_LIST[i], linewidth=1.0)
        ax.annotate(
            layer,
            xy=(0.01, 0.72),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )
        ax.spines[["right", "top"]].set_visible(False)

        if i < n_layers - 1:
            ax.tick_params(axis="x", bottom=False)
        else:
            ax.set_xlabel("Time (ms)", fontsize=15)

        if i == n_layers // 2:
            ax.set_ylabel(r"$\sum|I_\mathrm{syn}|$ (pA)", fontsize=13)

    plt.tight_layout()
    plt.savefig("figures/lfp_signal.png", dpi=300, bbox_inches="tight")


def plot_lfp_psd(
    currents,
    layer_order,
    dt=0.1,
    nperseg=None,
    noverlap=None,
    window="hann",
    max_freq_hz=100,
    f_min_hz=1.0,
):
    """
    Power spectral density of a biologically plausible LFP proxy via
    Welch's method.

    The LFP proxy is the sum of absolute synaptic currents over all neurons
    in the population at each recorded time step:

    .. math::

        \\text{LFP}(t) = \\sum_i \\left( |I_{\\text{ex},i}(t)| + |I_{\\text{in},i}(t)| \\right)

    This proxy was proposed and validated against electrode recordings in
    Mazzoni *et al.* (2015, PLOS Comput. Biol.); the underlying data are
    archived at https://doi.org/10.5061/dryad.j5r51.

    Welch's method divides the signal into overlapping segments, windows
    each one, computes a periodogram, and averages them.  Compared with a
    single-shot FFT this reduces variance in the PSD estimate and
    suppresses DC without requiring post-hoc smoothing or bin-skipping.

    Parameters
    ----------
    currents : dict[str, pd.DataFrame]
        Per-layer DataFrames with columns ``sender``, ``time_ms``,
        ``I_syn_ex``, ``I_syn_in``, as produced by
        :meth:`M1Network.save_current_data`.
    layer_order : list[str]
        Ordered list of layer names to plot.
    dt : float, optional
        Multimeter recording interval (ms).  Must match the ``interval``
        argument passed to :meth:`M1Network.create_current_recorders`.
        Default ``0.1``.
    nperseg : int or None, optional
        Number of samples per Welch segment.  Controls the trade-off
        between frequency resolution (``fs / nperseg`` Hz) and variance
        reduction (more segments = lower variance).  ``None`` (default)
        sets it to ``max(signal_length // 4, 64)`` at runtime, which
        targets ~7 averaged segments and scales with simulation length.
        For a 500 ms simulation at ``dt = 0.1`` ms this gives
        ``nperseg = 1 250`` → ~8 Hz resolution.
    noverlap : int or None, optional
        Number of samples shared between adjacent segments.  ``None``
        (default) uses 50 % overlap (``nperseg // 2``).
    window : str or array, optional
        Windowing function passed to :func:`scipy.signal.welch`.
        ``'hann'`` (default) minimises spectral leakage.
    max_freq_hz : float, optional
        Upper frequency limit of the plot (Hz).  Default ``100``.
    f_min_hz : float, optional
        Lower frequency limit of the plot (Hz); also used to mask the DC
        component.  Default ``1.0``.
    """
    fs = 1000.0 / dt  # sampling frequency in Hz

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, layer in enumerate(layer_order):
        data = currents[layer]

        # LFP proxy: Σ |I_syn_ex| + Σ |I_syn_in| per time step, over all neurons
        lfp_proxy = (
            data
            .assign(_abs=np.abs(data["I_syn_ex"]) + np.abs(data["I_syn_in"]))
            .groupby("time_ms")["_abs"]
            .sum()
            .sort_index()
            .values
        )

        # Resolve defaults at runtime so they scale with signal length
        _nperseg  = nperseg  if nperseg  is not None else max(len(lfp_proxy) // 4, 64)
        _noverlap = noverlap if noverlap is not None else _nperseg // 2

        freq, power = scipy.signal.welch(
            lfp_proxy,
            fs=fs,
            window=window,
            nperseg=_nperseg,
            noverlap=_noverlap,
            scaling="density",   # pA²/Hz
            detrend="constant",  # remove per-segment mean → suppresses DC
        )

        mask = freq >= f_min_hz
        ax.plot(freq[mask], power[mask], color=_COLOR_LIST[i], label=layer)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Frequency (Hz)", fontsize=15)
    ax.set_ylabel("PSD (pA²/Hz)", fontsize=15)
    ax.set_title(r"LFP Proxy PSD  $\left(\sum|I_\mathrm{syn}|\right)$", fontsize=15)
    ax.set_xlim(f_min_hz, max_freq_hz)
    ax.legend()
    plt.savefig("figures/lfp_psd.png", dpi=300, bbox_inches="tight")
