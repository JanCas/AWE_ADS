"""Compare isothermal (21 °C) vs heated-start cases (150 °C, various tau_thermal)."""
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import JansPlottingStuff as JPS

from awe_ads import run_wrapper, read_experiments
from utils.properties import BedProperties, EnvironmentalConditions, SorbentProperties, Isotherm

JPS.apply()
plt.rcParams.update({'font.size': 6})

env = EnvironmentalConditions(RH=0.65, T=21)
isotherm = Isotherm.read_from_file("utils/ev15_uptake.txt", env)

sorbent = SorbentProperties(
    particle_radius=1e-5,
    particle_density=1100 * 0.39,
    particle_diffusivity=1e-15,
    isotherm=isotherm,
    env=env,
    k_sorb_file="utils/ev15_kinetics.txt",
)

experiments = read_experiments()
final_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

# --- Define cases ---
common_bed = dict(
    sorbent_bed_height=1e-3,
    sorbent_bed_width=0.1,
    sorbent_bed_length=0.1,
    pore_diameter=4.51e-5,
    porosity=0.67,
)

cases = {
    "Isothermal 21°C": BedProperties(**common_bed, tau_thermal=1e6, T0=21 + 273),
    r"150°C, $\tau$=5 min": BedProperties(**common_bed, tau_thermal=5 * 60, T0=150 + 273),
    r"150°C, $\tau$=10 min": BedProperties(**common_bed, tau_thermal=10 * 60, T0=150 + 273),
    r"150°C, $\tau$=15 min": BedProperties(**common_bed, tau_thermal=15 * 60, T0=150 + 273),
}

# --- Run all cases ---
solutions = {}
for name, bed_prop in cases.items():
    print(f"\n=== Running: {name} ===")
    solutions[name] = run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent, final_time=final_time)


def total_moles(solution, bed_prop):
    n_vals = solution.ys.n.vals
    C_s_all = solution.ys.C_s.vals
    ny = n_vals.shape[1]
    dy = bed_prop.sorbent_bed_height / (ny - 1)

    w = jnp.ones(ny)
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)
    dV = w * dy * bed_prop.sorbent_bed_length * bed_prop.sorbent_bed_width

    sorbent_mass = dV * (1 - bed_prop.porosity) * sorbent.particle_density
    ads = jnp.sum(n_vals * sorbent_mass, axis=1)
    vap = jnp.sum(C_s_all * bed_prop.porosity * dV, axis=1)
    return np.array(ads + vap)


# --- Plot ---
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
ax_mol, ax_err, ax_temp = axes

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Experimental data (first experiment only)
exp = experiments[0]
t_exp = exp["ElapsedSeconds"].values
mol_exp = exp["mol_ads"].values
ax_mol.plot(t_exp / 3600, mol_exp, '--', color='grey', alpha=0.6, label="Experiment")

# Model results
for idx, (name, bed_prop) in enumerate(cases.items()):
    sol = solutions[name]
    ts = np.array(sol.ts) / 3600
    moles = total_moles(sol, bed_prop)
    T_arr = np.array(sol.ys.T) - 273

    ax_mol.plot(ts, moles, label=name, color=colors[idx])
    ax_temp.plot(ts, T_arr, label=name, color=colors[idx])

    # % error vs experiment
    model_interp = np.interp(t_exp, np.array(sol.ts), moles)
    pct_err = np.where(mol_exp > 1e-10, (model_interp - mol_exp) / mol_exp * 100, 0)
    ax_err.plot(t_exp / 3600, pct_err, color=colors[idx], label=name)

ax_mol.set_ylabel("Adsorbed [mol]")
ax_mol.set_title("Model vs Experiment")
ax_mol.legend(fontsize=7)

ax_err.set_ylabel("Error [%]")
ax_err.axhline(0, color='k', lw=0.5)
ax_err.set_ylim(-50, 50)
ax_err.legend(fontsize=7)

ax_temp.set_ylabel("Temperature [°C]")
ax_temp.set_xlabel("Time [h]")
ax_temp.legend(fontsize=7)

plt.tight_layout()
plt.show()
