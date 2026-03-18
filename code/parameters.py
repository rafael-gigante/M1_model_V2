import numpy as np
import nest

layers_name = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']
n_layer = [1148, 324, 268, 60, 1216, 304, 800, 164] # 9 columns

bg_layer = [2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850]

num_cols = 9

simulation_time = 1100.0 # ms

d_ex = 1.5      	# Excitatory delay (ms)
std_d_ex = 0.75 	# Std. Excitatory delay (ms)
d_in = 0.80      # Inhibitory delay (ms)
std_d_in = 0.4  	# Std. Inhibitory delay (ms)
tau_syn = 0.5    # Post-synaptic current time constant (ms)
tau_m   = 10.0		# membrane time constant (ms)
tau_ref = 2.0		# absolute refractory period (ms)
Cm      = 250.0		# membrane capacity (pF)
v_r     = -65.0		# reset potential (mV)
theta    = -50.0		# fixed firing threshold (mV)
w_ex = 87.8		   	# excitatory synaptic weight (pA)
std_w_ex = 0.1*w_ex     # standard deviation weigth (pA)
g = 4.0
bg_freq = 8.0

# LIF neuron parameters
neuron_params_LIF = {
    "C_m":        Cm,   # Membrane Capacitance (pF)
    "V_th":       theta,   # Threshold (mV)
    "t_ref":      tau_ref,     # Refractory Time (ms)
    "tau_m":      tau_m,    # Membrane Time Constant (ms)
    "V_reset":    v_r,   # Reset Value (mV)
    "E_L":        v_r,   # Resting Potential (Assumed same as Vr based on eq)
    "tau_syn_ex": tau_syn,     # Synaptic Time Constant (ms)
    "tau_syn_in": tau_syn,     # Synaptic Time Constant (ms)
    
    "I_e": 0.0,           # Constant input current
}