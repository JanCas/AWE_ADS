import pandas as pd
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Kvaerno5, TqdmProgressMeter
import jax.numpy as jnp
import jax
from equinox import Module
from utils.spatial_discretization import SpatialDiscretisation, SpatialDiscretisation2D
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions, Isotherm, AirFlow
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from jaxtyping import Float, Array

class BedState(Module):
    C_s: SpatialDiscretisation2D
    n: SpatialDiscretisation2D
    C_air: SpatialDiscretisation  # air channel concentration along x
    T: Float[Array, ""]          # lumped bed temperature [K]

#@jax.jit
def bed_ode(t, y: BedState, args):
    bed_props, sorbent, env, air_flow = args

    C_s = y.C_s
    n = y.n
    C_air = y.C_air.vals  # (nx,)

    T = 180+273
    rho_s = sorbent.particle_density
    porosity = bed_props.porosity
    D_vs = bed_props.bed_diffusivity(T)
    h_m = air_flow.h_m

    # LDF sorption (element-wise on 2D array)
    dndt = sorbent.k_sorb_C(C_s.vals) * (sorbent.isotherm(C_s.vals) - n.vals)

    # --- Bed: 2D Laplacian with convective BC coupled to air channel ---
    C = C_s.vals  # (ny, nx)
    dx, dy = C_s.δx, C_s.δy

    C_padded = jnp.pad(C, ((1, 1), (1, 1)), mode='edge')
    d2C_dy2 = (C_padded[2:, 1:-1] - 2*C + C_padded[:-2, 1:-1]) / dy**2
    d2C_dx2 = (C_padded[1:-1, 2:] - 2*C + C_padded[1:-1, :-2]) / dx**2
    laplacian = d2C_dx2 + d2C_dy2

    # Robin BC at top (row 0): coupled to local C_air(x), not uniform C_amb
    C_surface = C[0, :]
    d2C_dy2_top = 2*(C[1, :] - C_surface) / dy**2 + 2*h_m / (D_vs * dy) * (C_air - C_surface)
    laplacian = laplacian.at[0, :].set(d2C_dx2[0, :] + d2C_dy2_top)

    dcdt = D_vs * laplacian - (1 - porosity) / porosity * rho_s * dndt

    # --- Air channel: 1D plug flow with mass transfer to bed ---
    u = air_flow.flow_speed
    H_gap = air_flow.air_gap_height

    # Upwind advection: dC/dx ≈ (C[i] - C[i-1]) / dx, inlet at x=0 = C_amb
    C_air_upstream = jnp.roll(C_air, 1)
    C_air_upstream = C_air_upstream.at[0].set(env.C_amb)
    dCair_dx = (C_air - C_air_upstream) / dx

    dCair_dt = -u * dCair_dx - 2*h_m / H_gap * (C_air - C_surface)
    dCair_dt = dCair_dt.at[0].set(0)  # inlet fixed at C_amb

    # --- Lumped bed temperature ---
    dTdt = (env.T - T) / bed_props.tau_thermal

    return BedState(
        C_s=SpatialDiscretisation2D(C_s.x0, C_s.x_final, C_s.y0, C_s.y_final, dcdt),
        n=SpatialDiscretisation2D(n.x0, n.x_final, n.y0, n.y_final, dndt),
        C_air=SpatialDiscretisation(y.C_air.x0, y.C_air.x_final, dCair_dt),
        T=dTdt
    )

def run_wrapper(bed_props: BedProperties, env: EnvironmentalConditions, sorbent: SorbentProperties, air_flow: AirFlow, final_time: float):

    # 2D discretization: x (bed length) × y (bed height)
    nx = 20
    ny = 10

    # Initial conditions: bed empty, air channel at ambient
    C_s_vals = jnp.zeros((ny, nx))
    C_air_vals = jnp.zeros(nx).at[0].set(env.C_amb)  # inlet fixed at C_amb

    bed_state = BedState(
        C_s=SpatialDiscretisation2D(0, bed_props.sorbent_bed_length,
                                     0, bed_props.sorbent_bed_height, C_s_vals),
        n=SpatialDiscretisation2D(0, bed_props.sorbent_bed_length,
                                   0, bed_props.sorbent_bed_height, jnp.zeros((ny, nx))),
        C_air=SpatialDiscretisation(0, bed_props.sorbent_bed_length, C_air_vals),
        T=jnp.array(bed_props.T0)
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
        args=(bed_props, sorbent, env, air_flow),
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import JansPlottingStuff as JPS
    from plotting import plot_model_vs_experiment, animate_bed, plot_temperature

    JPS.apply()
    plt.rcParams.update({'font.size': 8})

    air_flow = AirFlow(
        flow_speed=5,
        air_gap_height=1.2e-3,
        h_m=8.0
    )

    bed_prop = BedProperties(
        sorbent_bed_height=1e-3,
        sorbent_bed_width=.1,
        sorbent_bed_length=.1,
        pore_diameter=4.51e-5,
        porosity=.67,
        tau_thermal=60*60,
        T0=180+273
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

    solution = run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent, air_flow=air_flow, final_time=final_time)

    plot_model_vs_experiment(solution, experiments, bed_prop, sorbent, env)
    # anim, anim_fig = animate_bed(solution, bed_prop)
    # anim.save('bed_animation.mp4', writer='ffmpeg', fps=30)
    # plt.close(anim_fig)
    plot_temperature(solution)
    plt.show()
