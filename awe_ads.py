import pandas as pd
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, Euler, PIDController, Tsit5, Kvaerno5, TqdmProgressMeter
import jax.numpy as jnp
import jax
from equinox import Module
import matplotlib.pyplot as plt
from utils.spatial_discretization import SpatialDiscretisation, SpatialDiscretisation2D
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions, Isotherm, rh_to_c, psat_water, IDEAL_GAS_CONST
import JansPlottingStuff as JPS
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path

jax.config.update("jax_enable_x64", True)

class BedState(Module):
    C_s: SpatialDiscretisation2D
    n: SpatialDiscretisation2D

@jax.jit
def bed_ode(t, y: BedState, args):
    bed_props, sorbent, env = args

    C_s = y.C_s
    n = y.n

    rho_s = sorbent.particle_density
    porosity = bed_props.porosity
    D_vs = bed_props.bed_diffusivity

    # LDF sorption (element-wise on 2D array)
    dndt = sorbent.k_sorb_C(C_s.vals) * (sorbent.isotherm(C_s.vals) - n.vals)

    # 2D Laplacian: no-flux on left/right/bottom, fixed on top
    C = C_s.vals  # (ny, nx)
    dx, dy = C_s.δx, C_s.δy

    # Pad with edge values → implements no-flux ghost nodes on all sides
    C_padded = jnp.pad(C, ((1, 1), (1, 1)), mode='edge')
    d2C_dy2 = (C_padded[2:, 1:-1] - 2*C + C_padded[:-2, 1:-1]) / dy**2
    d2C_dx2 = (C_padded[1:-1, 2:] - 2*C + C_padded[1:-1, :-2]) / dx**2
    laplacian = d2C_dx2 + d2C_dy2

    dcdt = D_vs * laplacian - (1 - porosity) / porosity * rho_s * dndt
    dcdt = dcdt.at[0, :].set(0)  # fixed concentration at top row (y = H)

    return BedState(
        C_s=SpatialDiscretisation2D(C_s.x0, C_s.x_final, C_s.y0, C_s.y_final, dcdt),
        n=SpatialDiscretisation2D(n.x0, n.x_final, n.y0, n.y_final, dndt)
    )

def run_wrapper(bed_props: BedProperties, env: EnvironmentalConditions, sorbent: SorbentProperties, final_time: float):

    # 2D discretization: x (bed length) × y (bed height)
    nx = 25
    ny = 25

    # Initial conditions: zero everywhere, top row (y=H) fixed at C_amb
    C_s_vals = jnp.zeros((ny, nx))
    C_s_vals = C_s_vals.at[0, :].set(env.C_amb)

    bed_state = BedState(
        C_s=SpatialDiscretisation2D(0, bed_props.sorbent_bed_length,
                                     0, bed_props.sorbent_bed_height, C_s_vals),
        n=SpatialDiscretisation2D(0, bed_props.sorbent_bed_length,
                                   0, bed_props.sorbent_bed_height, jnp.zeros((ny, nx)))
    )

    t0 = 0
    tf = final_time

    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)

    saveat = SaveAt(ts=jnp.linspace(0, tf, 10000))

    solution = diffeqsolve(
        terms=ODETerm(bed_ode),
        solver=Kvaerno5(),
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=tf,
        dt0=1e-3,
        y0=bed_state,
        args=(bed_props, sorbent, env),
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
    n_vals = solution.ys.n.vals  # shape: (n_timesteps, ny, nx)
    C_s_all = solution.ys.C_s.vals  # shape: (n_timesteps, ny, nx)

    ny, nx = n_vals.shape[1], n_vals.shape[2]
    dx = bed_props.sorbent_bed_length / (nx - 1)
    dy = bed_props.sorbent_bed_height / (ny - 1)

    # Trapezoidal weights: edges 1/2, corners 1/4
    w = jnp.ones((ny, nx))
    w = w.at[0, :].multiply(0.5)
    w = w.at[-1, :].multiply(0.5)
    w = w.at[:, 0].multiply(0.5)
    w = w.at[:, -1].multiply(0.5)
    dV = w * dx * dy * bed_props.sorbent_bed_width

    sorbent_mass_per_element = dV * (1 - bed_props.porosity) * sorbent.particle_density
    adsorbed_moles = jnp.sum(n_vals * sorbent_mass_per_element, axis=(1, 2))

    vapor_moles = jnp.sum(C_s_all * bed_props.porosity * dV, axis=(1, 2))

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

    ax1.plot(ts_arr / 3600, total_moles_arr, label="Model (total)")
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
    C_s_vals = np.array(C_s_all)  # shape: (n_timesteps, ny, nx)
    # dC/dy at top row for each x position
    dCdy_top = (C_s_vals[:, 0, :] - C_s_vals[:, 1, :]) / dy  # (n_timesteps, nx)
    # Integrate flux over x, times z-depth (bed_width)
    total_flux = bed_props.bed_diffusivity * np.sum(dCdy_top, axis=1) * dx * bed_props.sorbent_bed_width

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
        k_sorb_file="utils/ev15_kinetics.txt"
    )

    experiments = read_experiments()
    final_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

    solution = run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent, final_time=final_time)

    plot_model_vs_experiment(solution, experiments, bed_prop, sorbent)
