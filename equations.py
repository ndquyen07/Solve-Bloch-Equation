"""
Core equations for Semiconductor Bloch Equations.
"""

import numpy as np
from constants import *
from scipy.signal import spectrogram

def laser_pulse(t, chi0, delta_t):
    """Gaussian laser pulse theo đề bài.
    E(t) = (1/2) ℏ √π / δt * χ₀ * exp(-t²/δt²)
    """
    return 0.5 * hbar * np.sqrt(np.pi) / delta_t * chi0 * np.exp(-t**2 / delta_t**2)


def g(n, n1, Delta_epsilon):
    """Coulomb interaction function.
    g(n,n') = (1/√(n·Δε)) * ln[(√n + √n') / |√n - √n'|]
    """
    if n == n1:
        return 0.0

    sqrt_n = np.sqrt(n)
    sqrt_n1 = np.sqrt(n1)

    if abs(sqrt_n - sqrt_n1) < 1e-10:
        return 0.0

    ratio = (sqrt_n + sqrt_n1) / abs(sqrt_n - sqrt_n1)

    if ratio <= 0:
        return 0.0

    return 1.0 / np.sqrt(n * Delta_epsilon) * np.log(ratio)


# Precompute g matrix for efficiency
g_matrix = np.zeros((N, N))
for n in range(0, N):
    for n1 in range(0, N):
        g_matrix[n, n1] = g(n+1, n1+1, Delta_epsilon)


def compute_En(f_n, E_R, Delta_epsilon, g_matrix, use_coulomb=True):
    """
    Compute energy renormalization for each n theo đề bài.
    """
    
    if not use_coulomb:
        return np.zeros(N)
    
    En = np.zeros(N)
    for n in range(N):
        sum_term = np.sum(g_matrix[n, :] * 2 * f_n)
        En[n] = (np.sqrt(E_R) / np.pi) * Delta_epsilon * sum_term

    return np.nan_to_num(En, nan=0.0, posinf=0.0, neginf=0.0)


def compute_Omega_R(t, p_n, chi0, delta_t, E_R, Delta_epsilon, g_matrix, use_coulomb=True):
    """
    Compute Rabi frequency for each n.
    """

    laser = laser_pulse(t, chi0, delta_t)
    Omega = np.zeros(N, dtype=complex)
    
    if use_coulomb:
        for n in range(N):
            coulomb = (np.sqrt(E_R)/np.pi) * Delta_epsilon * np.sum(g_matrix[n, :] * p_n)
            Omega[n] = (laser + coulomb) / hbar
    else:
        Omega[:] = laser / hbar
    
    return np.nan_to_num(Omega, nan=0.0, posinf=0.0, neginf=0.0)


def compute_T2_time_dependent(f_n, T2_0, gamma):
    """
    Compute time-dependent T2 
    """

    # Tính mật độ N(t) = C0 * Σ √n f_n
    C0 = 2.0
    sqrt_n_array = np.sqrt(np.arange(1, N+1))
    N_total = C0 * np.sum(sqrt_n_array * f_n)
    
    # 1/T2 = 1/T2_0 + γ*N(t)
    T2_inv = 1.0 / T2_0 + gamma * N_total
    T2 = 1.0 / T2_inv if T2_inv > 0 else T2_0
    
    return T2


def bloch_equations(t, y, chi0, delta_t, Delta_0, T2_0, E_R, Delta_epsilon, g_matrix, use_time_dependent_T2=True, use_coulomb=True):
    """
    ∂f_n/∂t = -2 Im[Ω_n^R p_n*]                              
    ∂p_n/∂t = -i/ℏ [nΔε - Δ0 - E_n] p_n + i[1 - 2f_n] Ω_n^R - p_n/T2  
    """
    
    f_n = y[:N]
    p_n = y[N:2*N] + 1j * y[2*N:3*N]

    # Tính T2 phụ thuộc thời gian nếu cần
    if use_time_dependent_T2:
        T2 = compute_T2_time_dependent(f_n, T2_0, gamma)
    else:
        T2 = T2_0

    En = compute_En(f_n, E_R, Delta_epsilon, g_matrix, use_coulomb)
    Omega_R = compute_Omega_R(t, p_n, chi0, delta_t, E_R, Delta_epsilon, g_matrix, use_coulomb)

    df_dt = np.zeros(N)
    dp_dt = np.zeros(N, dtype=complex)

    for n in range(N):
        # ∂f_n/∂t = -2 Im[Ω_n^R p_n*]                     (0.5a)
        df_dt[n] = -2.0 * np.imag(Omega_R[n] * np.conj(p_n[n]))

        # ∂p_n/∂t = -i/ℏ [nΔε - Δ0 - E_n] p_n + i[1 - 2f_n] Ω_n^R - p_n/T2  (0.5b)
        detuning = (n+1) * Delta_epsilon - Delta_0 - En[n]
        dp_dt[n] = (-1j / hbar) * detuning * p_n[n] + 1j * (1.0 - 2.0 * f_n[n]) * Omega_R[n] - p_n[n] / T2

    # Pack into real array
    dy_dt = np.zeros(3*N)
    dy_dt[:N] = df_dt
    dy_dt[N:2*N] = np.real(dp_dt)
    dy_dt[2*N:3*N] = np.imag(dp_dt)

    return dy_dt


def calculate_absorption_spectrum(t, P_t, E_t):
    """
    Calculate absorption spectrum 
    α(ω) ∝ Im[P(ω)/E(ω)]
    """
    dt = t[1] - t[0]
    n = len(t)

    # Fourier Transform:
    # Sử dụng rời rạc hóa Riemann: Δt Σ_n f(t_n) e^(iωt_n)
    P_freq = np.fft.fft(P_t) * dt
    E_freq = np.fft.fft(E_t) * dt

    # Frequency axis
    freq = np.fft.fftfreq(n, dt)  # in 1/fs
    omega = 2 * np.pi * freq  # rad/fs

    # Take only positive frequencies
    pos_mask = freq > 0
    omega_pos = omega[pos_mask]
    P_freq_pos = P_freq[pos_mask]
    E_freq_pos = E_freq[pos_mask]
    
    # Absorption: α(ω) ∝ Im[P(ω)/E(ω)]
    # Tránh chia cho 0 bằng cách thêm epsilon nhỏ
    epsilon = 1e-10 * np.max(np.abs(E_freq_pos))
    alpha_raw = np.imag(P_freq_pos / (E_freq_pos + epsilon))
    
    # Lọc bỏ giá trị âm (nhiễu số ở vùng năng lượng thấp)
    # Phổ hấp thụ phải không âm về mặt vật lý
    alpha = np.maximum(alpha_raw, 0.0)

    return omega_pos, alpha