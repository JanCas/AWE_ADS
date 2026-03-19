import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import jax.numpy as jnp
from utils.properties import BedProperties, SorbentProperties, EnvironmentalConditions


def plot_model_vs_experiment(solution, experiments, bed_props: BedProperties, sorbent: SorbentProperties, env: EnvironmentalConditions):
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
    total_flux = bed_props.bed_diffusivity(env.T) * np.sum(dCdy_top, axis=1) * dx * bed_props.sorbent_bed_width

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


def plot_temperature(solution):
    ts = np.array(solution.ts)
    T = np.array(solution.ys.T)

    fig, ax = plt.subplots()
    ax.plot(ts / 3600, T - 273, label='Bed temperature')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title('Lumped Bed Temperature')
    ax.legend()
    plt.tight_layout()


def animate_bed(solution, bed_props: BedProperties, n_frames=200):
    ts = np.array(solution.ts)
    C_s_all = np.array(solution.ys.C_s.vals)  # (n_timesteps, ny, nx)
    n_all = np.array(solution.ys.n.vals)
    C_air_all = np.array(solution.ys.C_air.vals)  # (n_timesteps, nx)

    # Subsample frames evenly across time
    frame_idx = np.linspace(0, len(ts) - 1, n_frames, dtype=int)

    x_extent = [0, bed_props.sorbent_bed_length * 1e3]  # mm
    y_extent = [0, bed_props.sorbent_bed_height * 1e3]   # mm
    nx = C_s_all.shape[2]
    x_arr = np.linspace(x_extent[0], x_extent[1], nx)

    fig, (ax_air, ax_c, ax_n) = plt.subplots(3, 1, figsize=(12, 6),
                                              height_ratios=[1, 1, 1])

    # Fixed color/axis limits across the whole animation
    c_vmin, c_vmax = C_s_all.min(), C_s_all.max()
    n_vmin, n_vmax = n_all.min(), n_all.max()
    air_vmin, air_vmax = C_air_all.min(), C_air_all.max()

    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

    # Air channel: 1D line plot
    line_air, = ax_air.plot(x_arr, C_air_all[0], color='tab:blue')
    ax_air.set_xlim(x_extent)
    ax_air.set_ylim(air_vmin, air_vmax * 1.05)
    ax_air.set_ylabel(r'$C_{air}$ [mol/m³]')
    ax_air.set_xlabel('x [mm]')
    ax_air.set_title('Air channel concentration')

    # Bed C_s colormap
    im_c = ax_c.imshow(C_s_all[0], aspect='auto', origin='upper',
                        extent=extent,
                        vmin=c_vmin, vmax=c_vmax, cmap='viridis')
    fig.colorbar(im_c, ax=ax_c, label=r'$C_s$ [mol/m³]', location='right', shrink=0.9)
    ax_c.set_xlabel('x [mm]')
    ax_c.set_ylabel('y [mm]')

    # Bed n colormap
    im_n = ax_n.imshow(n_all[0], aspect='auto', origin='upper',
                        extent=extent,
                        vmin=n_vmin, vmax=n_vmax, cmap='inferno')
    fig.colorbar(im_n, ax=ax_n, label=r'$n$ [mol/kg]', location='right', shrink=0.9)
    ax_n.set_xlabel('x [mm]')
    ax_n.set_ylabel('y [mm]')

    title = fig.suptitle(f't = {ts[0]:.1f} s')

    def update(frame):
        idx = frame_idx[frame]
        line_air.set_ydata(C_air_all[idx])
        im_c.set_data(C_s_all[idx])
        im_n.set_data(n_all[idx])
        title.set_text(f't = {ts[idx]:.1f} s')
        return line_air, im_c, im_n, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    plt.tight_layout()
    return anim, fig
