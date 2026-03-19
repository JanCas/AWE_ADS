import pandas as pd
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, Euler, PIDController, Tsit5, Kvaerno5, TqdmProgressMeter
import jax.numpy as jnp
import jax
from equinox import Module
import matplotlib.pyplot as plt
from utils.spatial_discretization import SpatialDiscretisation
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions, Isotherm, rh_to_c, psat_water, IDEAL_GAS_CONST
import JansPlottingStuff as JPS
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path

jax.config.update("jax_enable_x64", True)

class BedState(Module):
    C_s: SpatialDiscretisation
    n: SpatialDiscretisation

@jax.jit
def bed_ode(t, y: BedState, args):
    bed_props, sorbent, env = args

    #extracting the parameters
    C_s = y.C_s
    n = y.n

    # Sorbent Properties
    k_sorb = sorbent.k_sorb
    rho_s = sorbent.particle_density

    # Geometry parameters
    porosity = bed_props.porosity

    # Sorbent Bed Diffusivity
    D_vs = bed_props.bed_diffusivity


    # LDF for sorption
    dndt = sorbent.k_sorb_C(C_s.vals) * (sorbent.isotherm(C_s.vals) - n.vals)
    #dndt = k_sorb * (sorbent.isotherm(C_s.vals) - n.vals)

    # Diffusion in the z direction
    C_lower = jnp.roll(C_s.vals, shift=1)   # C[i-1]
    C_upper = jnp.roll(C_s.vals, shift=-1)  # C[i+1]

    #Top BC (Fixed concentration)

    laplacian = (C_upper - 2*C_s.vals + C_lower) / C_s.δx**2
    # No-flux BC at z=0: use ghost node C_ghost = C[0], giving laplacian = (C[1] - C[0]) / dx²
    laplacian = laplacian.at[0].set((C_upper[0] - C_s.vals[0]) / C_s.δx**2)
    laplacian = laplacian.at[-1].set(0)  # Will be overridden by dcdt[-1] = 0 anyway

    dcdt =  D_vs*laplacian - (1-porosity)/porosity * rho_s * dndt
    dcdt = dcdt.at[-1].set(0)

    return BedState(
        C_s=SpatialDiscretisation(C_s.x0, C_s.x_final, dcdt),
        n=SpatialDiscretisation(n.x0, n.x_final, dndt)
    )

def run_wrapper(bed_props: BedProperties, env: EnvironmentalConditions, sorbent: SorbentProperties, final_time: float):

    #Discretization
    x0 = 0
    x_final = bed_props.sorbent_bed_height
    n = 25

    #initial conditions
    C_s_vals = jnp.zeros(n)
    C_s_vals  = C_s_vals.at[-1].set(env.C_amb)

    bed_state = BedState(
        C_s=SpatialDiscretisation(x0, x_final, C_s_vals),
        n = SpatialDiscretisation.discretise_fn(x0, x_final, n, lambda x: 0)
    )

    # temporal discretization
    t0 = 0
    tf = final_time

    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)

    saveat = SaveAt(ts=jnp.linspace(0, tf, 10000))

    solution = diffeqsolve(
        terms = ODETerm(bed_ode),
        solver = Kvaerno5(),
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=tf,
        dt0=1e-3,
        y0 = bed_state,
        args = (bed_props, sorbent, env),
        saveat=saveat,
        progress_meter=TqdmProgressMeter(),
        max_steps=int(1e7)
    )

    return solution


def read_experiments(folder="exp_data/cleaned"):
    folder = Path(folder)
    files = sorted(folder.glob("*_cleaned.csv"))
    experiments = []
    for i, f in enumerate(files):
        if i == 1:  # skip 2nd experiment
            continue
        df = pd.read_csv(f)
        df["mol_ads"] = df["Ads Weight"] / 420 / 18.01528
        experiments.append(df)
    return experiments


def plot_model_vs_experiment(solution, experiments, bed_props: BedProperties, sorbent: SorbentProperties):
    ts = solution.ts
    n_vals = solution.ys.n.vals  # shape: (n_timesteps, n_spatial_points)

    # Calculate total moles from model
    n_points = n_vals.shape[1]
    dz = bed_props.sorbent_bed_height / (n_points - 1)
    dV = bed_props.sorbent_bed_length * bed_props.sorbent_bed_width * dz
    sorbent_mass_per_element = dV * (1 - bed_props.porosity) * sorbent.particle_density
    adsorbed_moles = jnp.sum(n_vals * sorbent_mass_per_element, axis=1)

    # Vapor stored in pore space of the bed
    C_s_all = solution.ys.C_s.vals  # shape: (n_timesteps, n_spatial_points)
    vapor_moles = jnp.sum(C_s_all * bed_props.porosity * dV, axis=1)

    total_moles = adsorbed_moles + vapor_moles

    ts_arr = np.array(ts)
    total_moles_arr = np.array(total_moles)

    # --- Figure 1: Moles comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values
        ax1.plot(t_exp / 3600, mol_exp, label=f"Exp {i+1}")

        model_interp = np.interp(t_exp, ts_arr, total_moles_arr)
        pct_error = np.where(mol_exp != 0, (model_interp - mol_exp) / mol_exp * 100, 0)
        ax2.plot(t_exp / 3600, pct_error, label=f"Exp {i+1}")

    adsorbed_moles_arr = np.array(adsorbed_moles)
    ax1.plot(ts_arr / 3600, total_moles_arr, label="Model (total)")
    ax1.plot(ts_arr / 3600, adsorbed_moles_arr, label="Model (adsorbed)")
    ax1.set_ylabel("Ads Moles")
    ax1.set_title("Model vs Experiment")
    ax1.legend()

    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Error (%)")
    ax2.legend()

    plt.tight_layout()

    # --- Figure 2: Rate of adsorption comparison (5 min sampling) ---
    dt_sample = 120  # 2 minutes in seconds
    t_max = ts_arr[-1]
    t_sample = np.arange(0, t_max, dt_sample)

    # Model rate at sampled points
    model_moles_sampled = np.interp(t_sample, ts_arr, total_moles_arr)
    model_rate = np.diff(model_moles_sampled) / dt_sample
    t_rate_sample = (t_sample[:-1] + t_sample[1:]) / 2

    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True)

    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values

        # Resample experimental data at 5 min intervals
        exp_moles_sampled = np.interp(t_sample, t_exp, mol_exp)
        exp_rate = np.diff(exp_moles_sampled) / dt_sample

        ax3.plot(t_rate_sample / 3600, exp_rate, label=f"Exp {i+1}")

        # Rate error
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

    # --- Figure 3: Diffusive flux at top boundary ---
    C_s_vals = np.array(solution.ys.C_s.vals)  # shape: (n_timesteps, n_spatial_points)
    # One-sided difference: dC/dz at top ≈ (C[-1] - C[-2]) / dz
    dCdz_top = (C_s_vals[:, -1] - C_s_vals[:, -2]) / dz
    A_cross = bed_props.sorbent_bed_length * bed_props.sorbent_bed_width
    # Total diffusive flux into the bed (mol/s): positive = into bed (same sign as adsorption rate)
    total_flux = bed_props.bed_diffusivity * dCdz_top * A_cross

    # Resample diffusive flux at same 10 min intervals for comparison
    flux_sampled = np.interp(t_sample, ts_arr, total_flux)
    flux_rate = (flux_sampled[:-1] + flux_sampled[1:]) / 2  # avg flux over each interval

    fig3, ax5 = plt.subplots()

    ax5.plot(t_rate_sample / 3600, model_rate, label="Model Ads Rate")
    ax5.plot(t_rate_sample / 3600, flux_rate, label="Diffusive Flux")
    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values
        exp_moles_sampled = np.interp(t_sample, t_exp, mol_exp)
        exp_rate = np.diff(exp_moles_sampled) / dt_sample
        ax5.plot(t_rate_sample / 3600, exp_rate, label=f"Exp {i+1} Rate")

    ax5.set_xlabel("Time [h]")
    ax5.set_ylabel("Rate [mol/s]")
    ax5.set_title("Adsorption Rate vs Diffusive Flux")
    ax5.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    JPS.apply()
    plt.rcParams.update({'font.size': 8})

    bed_prop = BedProperties(
        sorbent_bed_height=1e-3,
        sorbent_bed_width=.1,
        sorbent_bed_length=.1,
        pore_diameter=4.51e-5,
        porosity=.67,
        vapor_diffusivity=2.25e-5
    )

    env = EnvironmentalConditions(
        RH=.65,
        T=21
    )

    isotherm = Isotherm.read_from_file("utils/ev15_uptake.txt", env)

    sorbent = SorbentProperties(
        particle_radius=1e-5,
        particle_density=1100 * .39,
        particle_diffusivity=1e-15,
        isotherm=isotherm,
        env=env,
        k_sorb_file="utils/D_mu_RH.txt"
    )

    experiments = read_experiments()
    final_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

    solution = run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent, final_time=final_time)

    plot_model_vs_experiment(solution, experiments, bed_prop, sorbent)
