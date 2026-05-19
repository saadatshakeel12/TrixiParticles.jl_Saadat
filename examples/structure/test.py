import numpy as np
import matplotlib.pyplot as plt

strain = np.linspace(0, 0.2, 100)
viscosity = 1e7  # Pa·s

strain_rates = [0.001, 1, 1000]
for rate in strain_rates:
    stress = viscosity * rate * np.ones_like(strain)  # constant stress for Newtonian
    plt.plot(strain, stress / 1e6, label=f"rate={rate} s$^{{-1}}$")

plt.xlabel("Strain")
plt.ylabel("Stress [MPa]")
plt.title("Viscous (no plasticity) Stress–Strain")
plt.legend()
plt.grid(True)
plt.savefig("viscous_stress_strain.png", dpi=150, bbox_inches='tight')