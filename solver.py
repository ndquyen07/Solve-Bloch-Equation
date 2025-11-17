"""
Numerical solver for ordinary differential equations.
"""

import numpy as np

def rk4(f, a, b, N_steps, y0, args):
    """Runge-Kutta 4th order method."""
    h = (b - a) / N_steps
    t = np.linspace(a, b, N_steps + 1)
    y = np.zeros((N_steps + 1, len(y0)))
    y[0] = y0

    for i in range(N_steps):
        k1 = h * f(t[i], y[i], *args)
        k2 = h * f(t[i] + h/2, y[i] + k1/2, *args)
        k3 = h * f(t[i] + h/2, y[i] + k2/2, *args)
        k4 = h * f(t[i] + h, y[i] + k3, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y