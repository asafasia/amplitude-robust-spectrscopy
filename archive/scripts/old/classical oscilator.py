import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
m = 1.0  # mass
gamma = 0.2  # damping coefficient
k = 2.0  # spring constant
F0 = 1.0  # amplitude of the driving force


# Define the driving force
def F(t, omega):
    return F0 * np.cos(omega * t)


# Define the differential equation
def damped_driven_oscillator(t, y, omega):
    x, v = y
    dxdt = v
    dvdt = (F(t, omega) - gamma * v - k * x) / m
    return [dxdt, dvdt]


# Function to compute the steady-state amplitude
def get_steady_state_amplitude(omega):
    # Initial conditions: x(0) = 0, v(0) = 0
    y0 = [0.0, 0.0]

    # Time span for the solution
    t_span = (0, 200)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve the differential equation
    solution = solve_ivp(damped_driven_oscillator, t_span, y0, t_eval=t_eval, args=(omega,))

    # Extract the displacement x
    x = solution.y[0]

    # Consider the steady-state part of the solution (last 10% of the time points)
    steady_state_x = x[int(0.9 * len(x)):]

    # Calculate the amplitude as the max displacement in the steady state
    amplitude = np.max(np.abs(steady_state_x))
    return amplitude


# Define the range of driving frequencies
omega_values = np.linspace(0.1, 2.0, 501)

# Calculate the steady-state amplitudes for each driving frequency
amplitudes = [get_steady_state_amplitude(omega) for omega in omega_values]

# Plot the frequency response
plt.figure(figsize=(10, 5))
plt.plot(omega_values, amplitudes, label='Amplitude')
plt.xlabel('Driving Frequency (omega)')
plt.ylabel('Steady-State Amplitude')
plt.title('Frequency Response of Damped Driven Harmonic Oscillator')
plt.legend()
plt.grid()
plt.show()