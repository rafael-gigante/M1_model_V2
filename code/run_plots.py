import plots as pl
import pandas as pd
import parameters as params

sim_time = params.simulation_time # Total simulation time in ms

# Read the combined spike data
df_l23e = pd.read_csv('data/combined_spikes_23E.csv')
df_l23i = pd.read_csv('data/combined_spikes_23I.csv')
df_l4e = pd.read_csv('data/combined_spikes_4E.csv')
df_l4i = pd.read_csv('data/combined_spikes_4I.csv')
df_l5e = pd.read_csv('data/combined_spikes_5E.csv')
df_l5i = pd.read_csv('data/combined_spikes_5I.csv')
df_l6e = pd.read_csv('data/combined_spikes_6E.csv')
df_l6i = pd.read_csv('data/combined_spikes_6I.csv')

spikes = {
    '2/3E': df_l23e[df_l23e['time_ms'] > sim_time-500.0],
    '2/3I': df_l23i[df_l23i['time_ms'] > sim_time-500.0],
    '4E': df_l4e[df_l4e['time_ms'] > sim_time-500.0],
    '4I': df_l4i[df_l4i['time_ms'] > sim_time-500.0],
    '5E': df_l5e[df_l5e['time_ms'] > sim_time-500.0],
    '5I': df_l5i[df_l5i['time_ms'] > sim_time-500.0],
    '6E': df_l6e[df_l6e['time_ms'] > sim_time-500.0],
    '6I': df_l6i[df_l6i['time_ms'] > sim_time-500.0]}

layer_order = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']
layer_colors = {
    "2/3E": "#8ca465",    "2/3I": "#dec47c",
    "4E":   "#487a99", "4I":   "#d19f7f",
    "5E":   "#385f49", "5I":   "#ae5b5e",
    "6E":   "#2e4876",  "6I":   "#85588c"
}


#pl.raster_plot(spikes, sim_time, layer_order, layer_colors)
#pl.plot_rates(spikes, layer_order, sim_time)
#pl.CV_boxplot(spikes, layer_order, method='CV')
#pl.CV_boxplot(spikes, layer_order, method='CV2')
#pl.FR_boxplot(spikes, layer_order)
#pl.kuramoto_order_boxplot(spikes, layer_order)
#pl.sttc_boxplot(spikes, layer_order)
pl.plot_psd(spikes, layer_order, sim_time)