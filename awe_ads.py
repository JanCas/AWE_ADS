import pandas as pd
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, Euler, PIDController, Tsit5, Kvaerno5, TqdmProgressMeter
import jax.numpy as jnp
import jax
from equinox import Module
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.spatial_discretization import SpatialDiscretisation
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions, Isotherm, rh_to_c, psat_water, IDEAL_GAS_CONST
import JansPlottingStuff as JPS
import numpy as np

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
    #dndt = sorbent.k_sorb_C(C_s.vals) * (sorbent.isotherm(C_s.vals) - n.vals)
    dndt = k_sorb * (sorbent.isotherm(C_s.vals) - n.vals)

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

def run_wrapper(bed_props: BedProperties, env: EnvironmentalConditions, sorbent: SorbentProperties):

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
    tf = int(7 * 3600)
    # dt = 1e-3

    # stepsize_controller = ConstantStepSize()
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)

    # Saveat setup
    # saveat = SaveAt(t0=True, steps=True)
    saveat = SaveAt(ts=jnp.linspace(0, tf, 10000))  # only save 500 points

    solution = diffeqsolve(
        terms = ODETerm(bed_ode),
        # solver = Euler(),
        # solver = Tsit5(),
        solver = Kvaerno5(),  # implicit solver for stiff diffusion
        stepsize_controller=stepsize_controller,
        t0=t0,
        t1=tf,
        # dt0=dt,
        dt0=1e-3,  # initial guess, will adapt
        y0 = bed_state,
        args = (bed_props, sorbent, env),
        saveat=saveat,
        progress_meter=TqdmProgressMeter(),
        # max_steps=int(tf/dt)+20
        max_steps=int(1e7)
    )


    return solution


def plot_total_n(solution, bed_props: BedProperties, sorbent: SorbentProperties):
    ts = solution.ts
    n_vals = solution.ys.n.vals  # shape: (n_timesteps, n_spatial_points)

    # Calculate volume element for each spatial point (discretization is over height)
    n_points = n_vals.shape[1]
    dz = bed_props.sorbent_bed_height / (n_points - 1)
    dV = bed_props.sorbent_bed_length * bed_props.sorbent_bed_width * dz

    # Mass of sorbent per element = volume × (1-porosity) × particle_density
    sorbent_mass_per_element = dV * (1 - bed_props.porosity) * sorbent.particle_density

    # Total moles = sum(n × mass) for each timestep
    total_moles = jnp.sum(n_vals * sorbent_mass_per_element, axis=1)

    np.savetxt("test.txt", np.column_stack((ts, total_moles)))
    
    x = pd.read_csv("exp_ads_data.csv")
    x["mol_ads"] = x["ads_mass(g)"] / 420 / 18.01528


    # Interpolate model onto experimental time grid to compute error
    model_interp = np.interp(x["Time(s)"].values, np.array(ts), np.array(total_moles))
    error = model_interp - x["mol_ads"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(x["Time(s)"]/3600, x["mol_ads"], color='r', label="exp")
    ax1.plot(ts / 3600, total_moles, label="model")
    ax1.set_ylabel("Total moles captured")
    ax1.set_title("Total Moles Captured vs Time")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x["Time(s)"]/3600, error, color='k')
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Error (model - exp)")
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_n_profiles(solution, bed_props: BedProperties, env: EnvironmentalConditions):
    ts = solution.ts
    n_vals = solution.ys.n.vals  # shape: (n_timesteps, n_spatial_points)
    C_s_vals = solution.ys.C_s.vals
    n_points = n_vals.shape[1]
    z = jnp.linspace(0, bed_props.sorbent_bed_height * 1000, n_points)  # height in mm

    # Convert concentration to RH: C = RH * Psat / (R * T) => RH = C * R * T / Psat
    RH_vals = C_s_vals * IDEAL_GAS_CONST * env.T / psat_water(env.T)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    for i in range(n_points):
        ax1.plot(ts / 3600, n_vals[:, i], label=f"z = {z[i]:.3f} mm")
    ax1.set_ylabel("Sorbent loading (n)")
    # ax1.legend()
    ax1.grid(True)

    for i in range(n_points):
        ax2.plot(ts / 3600, C_s_vals[:, i], label=f"z = {z[i]:.3f} mm")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Concentration C_s")
    # ax2.legend()
    ax2.grid(True)

    for i in range(n_points):
        ax3.plot(ts / 3600, RH_vals[:, i], label=f"z = {z[i]:.3f} mm")
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("RH")
    # ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Custom plotting style library (Designed by me)
    JPS.apply()

    bed_prop = BedProperties(
        sorbent_bed_height=1e-3,
        sorbent_bed_width=.1,
        sorbent_bed_length=.1,
        pore_diameter=4.51e-5,
        porosity=.67,
        vapor_diffusivity=2.25e-5
    )

    env = EnvironmentalConditions(
        RH=.62,
        T=21
    )

    isotherm = Isotherm.read_from_file("utils/ev15_uptake.txt", env)

    sorbent = SorbentProperties(
        particle_radius=1e-5,
        particle_density=1100 * .38,
        particle_diffusivity=1e-15,
        isotherm=isotherm,
        env=env,
        k_sorb_file="utils/D_mu_RH - Copy.txt"
    )

    solution=run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent)

    plot_total_n(solution, bed_prop, sorbent)
    plot_n_profiles(solution, bed_prop, env)
