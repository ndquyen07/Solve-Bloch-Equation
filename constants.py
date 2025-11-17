"""
Constants and parameters for Semiconductor Bloch Equations simulation.
"""

# Physical constants
hbar = 658.5  # meV·fs

# Simulation parameters theo đề bài
chi0_values = [0.1, 1.0, 2.0]
delta_t = 25.0         # fs (laser pulse width)
Delta_0 = 30.0         # meV (detuning)
dt = 2.0              # fs (time step)
t_max = 500.0         # fs (maximum time)
t_0 = -3 * delta_t    # fs (initial time)
N = 100               # Number of energy levels
epsilon_max = 300.0   # meV (maximum energy level)
T2 = 200.0            # fs (dephasing time) 
T2_0 = 210.0          # fs (initial dephasing time)
E_R = 1.0             # meV (Rydberg energy)
Delta_epsilon = epsilon_max / N  # meV (energy spacing)

gamma = 6.5e-20       # cm³/fs (scattering coefficient)
