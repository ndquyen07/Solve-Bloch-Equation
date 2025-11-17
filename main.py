"""
Main execution script for Semiconductor Bloch Equations simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from constants import *
from equations import bloch_equations, calculate_absorption_spectrum, g_matrix, laser_pulse
from solver import rk4

def run_simulation(chi0, use_time_dependent_T2=True, use_coulomb=True):
    """
    Run the simulation for a given chi0 value.
    """

    coulomb_str = "with Coulomb" if use_coulomb else "without Coulomb"
    print(f"\nRunning simulation for χ₀ = {chi0} ({coulomb_str})")
    if use_time_dependent_T2:
        print(f"  Using time-dependent T2(t) with T2_0={T2_0}fs, γ={gamma}")
    else:
        print(f"  Using constant T2={T2_0}fs")

    # Initial conditions:
    # f_n(t0) = 0, p_n(t0) = 0
    y0 = np.zeros(3*N)
    N_steps = int((t_max - t_0) / dt)

    # Solve using RK4
    args = (chi0, delta_t, Delta_0, T2_0, E_R, Delta_epsilon, g_matrix, use_time_dependent_T2, use_coulomb)
    t, y = rk4(bloch_equations, t_0, t_max, N_steps, y0, args)

    # Extract results
    f_n = y[:, :N]
    p_n_real = y[:, N:2*N]
    p_n_imag = y[:, 2*N:3*N]
    p_n = p_n_real + 1j * p_n_imag

    # Compute observables
    C0 = 2.0
    sqrt_n_array = np.sqrt(np.arange(1, N+1))  
    N_t = C0 * np.sum(sqrt_n_array * f_n, axis=1)
    
    # Total polarization
    P_total_complex = np.sum(p_n, axis=1)
    P_t = np.abs(P_total_complex)

    # Laser field E(t)
    E_t = np.array([laser_pulse(ti, chi0, delta_t) for ti in t])

    # Calculate absorption spectrum: α(ω) ∝ Im[P(ω)/E(ω)]
    omega_pos, alpha = calculate_absorption_spectrum(t, P_total_complex, E_t)
    energy = hbar * omega_pos


    print(f"Simulation completed for χ₀ = {chi0} ({coulomb_str})")
    return t, N_t, P_t, f_n, p_n, energy, alpha


def main():
    """Main function to run simulations for all chi0 values.
    """

    results = {}
    
    for chi0 in chi0_values:
        # Simulation với tương tác Coulomb (Hartree-Fock)
        results[f'{chi0}_coulomb'] = run_simulation(chi0, use_time_dependent_T2=True, use_coulomb=True)
        
        # Simulation không có tương tác Coulomb (free particle)
        results[f'{chi0}_no_coulomb'] = run_simulation(chi0, use_time_dependent_T2=True, use_coulomb=False)
        
    


if __name__ == "__main__":
    main()