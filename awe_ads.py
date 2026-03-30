import pandas as pd
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Kvaerno5, TqdmProgressMeter
import jax.numpy as jnp
import jax
from equinox import Module
from utils.spatial_discretization import SpatialDiscretisation
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions, Isotherm, rh_to_c
from pathlib import Path

jax.config.update("jax_enable_x64", True)


class BedState(Module):
    C_s: SpatialDiscretisation   # gas-phase concentration along y
    n: SpatialDiscretisation     # adsorbed amount along y


def bed_ode(t, y: BedState, args):
    bed_props, sorbent, env = args

    C_s = y.C_s
    n = y.n
    C = C_s.vals  # (ny,)

    rho_s = sorbent.particle_density
    porosity = bed_props.porosity
    D_vs = bed_props.bed_diffusivity(env.T)

    # LDF sorption with loading-dependent k
    dndt = sorbent.k_ldf(n.vals) * (sorbent.isotherm(C) - n.vals)

    # --- 1D diffusion in y with constant C at top boundary ---
    dy = C_s.δx
    ny = C.shape[0]

    # Pad with ghost nodes: top (index 0) uses Dirichlet, bottom uses zero-flux
    C_top_ghost = 2 * env.C_amb - C[0]
    C_bot_ghost = C[-1]

    C_padded = jnp.concatenate([C_top_ghost[None], C, C_bot_ghost[None]])
    d2C_dy2 = (C_padded[2:] - 2 * C + C_padded[:-2]) / dy**2

    dcdt = D_vs * d2C_dy2 - (1 - porosity) / porosity * rho_s * dndt

    return BedState(
        C_s=SpatialDiscretisation(C_s.x0, C_s.x_final, dcdt),
        n=SpatialDiscretisation(n.x0, n.x_final, dndt),
    )


def run_wrapper(bed_props: BedProperties, env: EnvironmentalConditions, sorbent: SorbentProperties, final_time: float):

    ny = 20

    # Initial conditions
    C_init = rh_to_c(0.05, env.T)
    n0_val = sorbent.isotherm(C_init)  # equilibrium loading at 5% RH

    bed_state = BedState(
        C_s=SpatialDiscretisation(0, bed_props.sorbent_bed_height, jnp.zeros(ny)),
        n=SpatialDiscretisation(0, bed_props.sorbent_bed_height, jnp.full(ny, n0_val)),
    )

    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)
    saveat = SaveAt(ts=jnp.linspace(0, final_time, 10000))

    solution = diffeqsolve(
        terms=ODETerm(bed_ode),
        solver=Kvaerno5(),
        stepsize_controller=stepsize_controller,
        t0=0,
        t1=final_time,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import JansPlottingStuff as JPS
    from plotting import plot_model_vs_experiment

    JPS.apply()
    plt.rcParams.update({'font.size': 8})

    bed_prop = BedProperties(
        sorbent_bed_height=1e-3,
        sorbent_bed_width=.1,
        sorbent_bed_length=.1,
        pore_diameter=4.51e-5,
        porosity=.67,
    )

    env = EnvironmentalConditions(
        RH=.65,
        T=21
    )

    isotherm = Isotherm.read_from_file("utils/ev15_uptake.txt", env)

    sorbent = SorbentProperties(
        particle_density=1100 * .45,
        isotherm=isotherm,
    )

    experiments = read_experiments()
    final_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

    solution = run_wrapper(bed_props=bed_prop, env=env, sorbent=sorbent, final_time=final_time)

    plot_model_vs_experiment(solution, experiments, bed_prop, sorbent, env)
    plt.show()
