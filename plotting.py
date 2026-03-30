import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions


def plot_model_vs_experiment(solution, experiments, bed_props: BedProperties, sorbent: SorbentProperties, env: EnvironmentalConditions):
    ts = solution.ts
    n_vals = solution.ys.n.vals  # shape: (n_timesteps, ny)
    C_s_all = solution.ys.C_s.vals  # shape: (n_timesteps, ny)

    ny = n_vals.shape[1]
    dy = bed_props.sorbent_bed_height / (ny - 1)

    # Trapezoidal weights along y
    w = jnp.ones(ny)
    w = w.at[0].multiply(0.5)
    w = w.at[-1].multiply(0.5)
    dV = w * dy * bed_props.sorbent_bed_length * bed_props.sorbent_bed_width

    sorbent_mass_per_element = dV * (1 - bed_props.porosity) * sorbent.particle_density
    adsorbed_moles = jnp.sum(n_vals * sorbent_mass_per_element, axis=1)
    vapor_moles = jnp.sum(C_s_all * bed_props.porosity * dV, axis=1)
    total_moles = adsorbed_moles + vapor_moles

    ts_arr = np.array(ts)
    total_moles_arr = np.array(total_moles)
    total_moles_arr = total_moles_arr - total_moles_arr[0]  # zero-shift to start from 0

    # --- Figure 1: Moles comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values
        ax1.plot(t_exp / 3600, mol_exp, label=f"Exp {i+1}")

        model_interp = np.interp(t_exp, ts_arr, total_moles_arr)
        pct_error = np.where(mol_exp != 0, (model_interp - mol_exp) / mol_exp * 100, 0)
        ax2.plot(t_exp / 3600, pct_error, label=f"Exp {i+1}")

    ax1.plot(ts_arr / 3600, total_moles_arr, label="Model (total)")
    ax1.set_ylabel("Ads Moles")
    ax1.set_title("Model vs Experiment")
    ax1.legend()

    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Error (%)")
    ax2.legend()

    plt.tight_layout()

    # --- Figure 2: Rate of adsorption comparison (2 min sampling) ---
    dt_sample = 120  # 2 minutes in seconds
    t_max = ts_arr[-1]
    t_sample = np.arange(0, t_max, dt_sample)

    model_moles_sampled = np.interp(t_sample, ts_arr, total_moles_arr)
    model_rate = np.diff(model_moles_sampled) / dt_sample
    t_rate_sample = (t_sample[:-1] + t_sample[1:]) / 2

    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True)

    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values

        exp_moles_sampled = np.interp(t_sample, t_exp, mol_exp)
        exp_rate = np.diff(exp_moles_sampled) / dt_sample

        ax3.plot(t_rate_sample / 3600, exp_rate, label=f"Exp {i+1}")

        rate_error = np.where(exp_rate != 0, (model_rate[:len(exp_rate)] - exp_rate) / exp_rate * 100, 0)
        ax4.plot(t_rate_sample[:len(rate_error)] / 3600, rate_error, label=f"Exp {i+1}")

    ax3.plot(t_rate_sample / 3600, model_rate, label="Model")
    ax3.set_ylabel("Rate [mol/s]")
    ax3.set_title("Rate of Adsorption (2 min sampling)")
    ax3.legend()

    ax4.set_xlabel("Time [h]")
    ax4.set_ylabel("Rate Error (%)")
    ax4.legend()

    plt.tight_layout()
