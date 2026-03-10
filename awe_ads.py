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
    total_moles = jnp.sum(n_vals * sorbent_mass_per_element, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot individual experiments and their pct error vs model
    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values
        ax1.plot(t_exp / 3600, mol_exp, label=f"Exp {i+1}")

        model_interp = np.interp(t_exp, np.array(ts), np.array(total_moles))
        pct_error = np.where(mol_exp != 0, (model_interp - mol_exp) / mol_exp * 100, 0)
        ax2.plot(t_exp / 3600, pct_error, label=f"Exp {i+1}")

    # Plot model
    ax1.plot(np.array(ts) / 3600, np.array(total_moles), label="Model")
    ax1.set_ylabel("Ads Moles")
    ax1.set_title("Model vs Experiment")
    ax1.legend()

    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Error (%)")
    ax2.legend()

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
