# thermo-active-VSA

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# Kelvin-Voigt Model: Stress-Strain Relationship
def kelvin_voigt_model(epsilon, E1, eta):
    """
    Computes stress response for given strain using the Kelvin-Voigt model.
    sigma(t) = E1 * epsilon(t) + eta * d(epsilon)/dt
    """
    d_epsilon_dt = np.gradient(epsilon, dx=dt)
    sigma = E1 * epsilon + eta * d_epsilon_dt
    return sigma

# Generate Synthetic Strain Data
t_max = 10  # seconds
dt = 0.1
t = np.arange(0, t_max, dt)
epsilon = 0.05 * (1 - np.exp(-0.5 * t))  # Example strain evolution

# Ground Truth Parameters (for synthetic dataset)
E1_true = 500  # Elastic modulus in Pascals
eta_true = 50   # Viscosity coefficient in Pascal-seconds

# Generate Synthetic Stress Data
sigma_true = kelvin_voigt_model(epsilon, E1_true, eta_true)

# Add Noise to Simulate Real Data
sigma_noisy = sigma_true + np.random.normal(scale=5, size=len(sigma_true))

# System Identification: Least Squares Fit
def fit_kelvin_voigt(epsilon, E1, eta):
    return kelvin_voigt_model(epsilon, E1, eta)

# Fit Model Parameters to Noisy Data
popt, _ = curve_fit(fit_kelvin_voigt, epsilon, sigma_noisy, p0=[100, 10])
E1_est, eta_est = popt

# Simulated Response Using Identified Parameters
sigma_est = kelvin_voigt_model(epsilon, E1_est, eta_est)

# Plot Results
plt.figure(figsize=(8, 5))
plt.plot(t, sigma_true, label="True Stress (Ground Truth)", linestyle='dashed', color='black')
plt.plot(t, sigma_noisy, label="Noisy Observed Stress", alpha=0.6, color='red')
plt.plot(t, sigma_est, label=f"Estimated Stress (E1={E1_est:.2f}, η={eta_est:.2f})", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.legend()
plt.title("Kelvin-Voigt Model System Identification")
plt.grid()
plt.show()

# Save Results for GitHub Sharing
np.savetxt("kelvin_voigt_results.csv", np.column_stack((t, epsilon, sigma_noisy, sigma_est)), delimiter=",", 
           header="Time,Strain,Observed_Stress,Estimated_Stress", comments="")

# Print Identified Parameters
print(f"Identified Parameters: E1 = {E1_est:.2f} Pa, η = {eta_est:.2f} Pa.s")

