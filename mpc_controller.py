"""
Simple MPC controller for the 1D sorbent bed.

Control input:  active ∈ [0, 1]  — gates C_amb at the top Dirichlet BC.
Objective:      minimize  -total_adsorbed (terminal)  +  penalty * active (stage)

When the adsorption rate drops low enough that the marginal gain over the
prediction horizon no longer outweighs the running penalty, the MPC drives
active → 0 and terminates the cycle.
"""

import do_mpc
import casadi as cas
import numpy as np
from mpc_awe_ads import (
    bed_diffusivity, _load_isotherm, _load_k_sorb, rh_to_c, read_experiments,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model (with control input)
# ──────────────────────────────────────────────────────────────────────────────
def build_mpc_model(ny, isotherm_interp, k_sorb_interp, bed_length, bed_width):
    model = do_mpc.model.Model("continuous", "SX")

    # States
    C = model.set_variable("_x", "C", shape=(ny, 1))
    n = model.set_variable("_x", "n", shape=(ny, 1))
    T = model.set_variable("_x", "T")

    # Control input: adsorption on/off
    active = model.set_variable("_u", "active")

    # Parameters
    porosity    = model.set_variable("_p", "porosity")
    rho_s       = model.set_variable("_p", "rho_s")
    d_pore      = model.set_variable("_p", "d_pore")
    tau_thermal = model.set_variable("_p", "tau_thermal")
    bed_height  = model.set_variable("_p", "bed_height")

    # Time-varying parameters
    C_amb = model.set_variable("_tvp", "C_amb")
    T_env = model.set_variable("_tvp", "T_env")

    # Grid
    dy = bed_height / (ny - 1)

    # Diffusivity
    D_vs = bed_diffusivity(T, porosity, d_pore)

    # Sorption
    isotherm_map = isotherm_interp.map(ny)
    k_sorb_map   = k_sorb_interp.map(ny)
    n_eq = isotherm_map(C.T).T
    k_s  = k_sorb_map(C.T).T
    dndt = k_s * (n_eq - n)

    # Diffusion — top BC gated by active
    C_amb_eff = active * C_amb
    C_top_ghost = 2 * C_amb_eff - C[0]
    C_bot_ghost = C[ny - 1]
    C_padded = cas.vertcat(C_top_ghost, C, C_bot_ghost)
    d2C_dy2 = (C_padded[2:ny + 2] - 2 * C + C_padded[0:ny]) / dy ** 2

    dCdt = D_vs * d2C_dy2 - (1 - porosity) / porosity * rho_s * dndt
    dTdt = (T_env - T) / tau_thermal

    model.set_rhs("C", dCdt)
    model.set_rhs("n", dndt)
    model.set_rhs("T", dTdt)

    # Auxiliary: total adsorbed moles (for objective)
    w = np.ones((ny, 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    dV = w * dy * bed_length * bed_width
    sorbent_mass_elem = dV * (1 - porosity) * rho_s
    total_n = cas.sum1(n * sorbent_mass_elem)
    model.set_expression("total_n", total_n)

    model.setup()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# MPC controller
# ──────────────────────────────────────────────────────────────────────────────
def build_mpc(model, params, n_horizon=20, t_step=60.0, penalty=1e-5):
    """
    penalty : cost per second of running the adsorption.
              Increase → earlier termination.  Decrease → longer adsorption.
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_horizon = n_horizon
    mpc.settings.t_step = t_step
    mpc.settings.n_robust = 0
    mpc.settings.store_full_solution = True
    mpc.settings.supress_ipopt_output()

    # Objective: minimize  -total_n (terminal)  +  penalty * active (stage)
    mterm = -model.aux["total_n"]
    lterm = penalty * model.u["active"]
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Small penalty on input change to discourage chattering
    mpc.set_rterm(active=1e-4)

    # Bounds
    mpc.bounds["lower", "_u", "active"] = 0.0
    mpc.bounds["upper", "_u", "active"] = 1.0

    # Parameter and TVP functions
    p_template = mpc.get_p_template(1)
    p_template["_p", :, "porosity"]    = params["porosity"]
    p_template["_p", :, "rho_s"]       = params["rho_s"]
    p_template["_p", :, "d_pore"]      = params["d_pore"]
    p_template["_p", :, "tau_thermal"] = params["tau_thermal"]
    p_template["_p", :, "bed_height"]  = params["bed_height"]

    def p_fun(t_now):
        return p_template
    mpc.set_p_fun(p_fun)

    tvp_template = mpc.get_tvp_template()
    tvp_template["_tvp", :, "C_amb"] = params["C_amb"]
    tvp_template["_tvp", :, "T_env"] = params["T_env"]

    def tvp_fun(t_now):
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()
    return mpc


# ──────────────────────────────────────────────────────────────────────────────
# Simulator (with control input)
# ──────────────────────────────────────────────────────────────────────────────
def build_simulator(model, params, t_step=60.0):
    simulator = do_mpc.simulator.Simulator(model)

    simulator.settings.t_step = t_step
    simulator.settings.abstol = 1e-10
    simulator.settings.reltol = 1e-8

    p_template = simulator.get_p_template()
    p_template["porosity"]    = params["porosity"]
    p_template["rho_s"]       = params["rho_s"]
    p_template["d_pore"]      = params["d_pore"]
    p_template["tau_thermal"] = params["tau_thermal"]
    p_template["bed_height"]  = params["bed_height"]

    def p_fun(t_now):
        return p_template
    simulator.set_p_fun(p_fun)

    tvp_template = simulator.get_tvp_template()
    tvp_template["C_amb"] = params["C_amb"]
    tvp_template["T_env"] = params["T_env"]

    def tvp_fun(t_now):
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    ny = params["ny"]
    simulator.x0["C"] = np.zeros((ny, 1))
    simulator.x0["n"] = np.zeros((ny, 1))
    simulator.x0["T"] = params["T0"]

    return simulator


# ──────────────────────────────────────────────────────────────────────────────
# Closed-loop run
# ──────────────────────────────────────────────────────────────────────────────
def run_mpc(params, max_time, t_step=60.0, n_horizon=20, penalty=1e-5):
    ny = params["ny"]

    isotherm_interp = _load_isotherm(
        params["isotherm_file"], params["T_env"], params.get("RH", 0.65)
    )
    k_sorb_interp = _load_k_sorb(
        params["kinetics_file"], params["T_env"], params.get("RH", 0.65)
    )

    model = build_mpc_model(
        ny, isotherm_interp, k_sorb_interp,
        params["bed_length"], params["bed_width"],
    )

    mpc = build_mpc(model, params, n_horizon=n_horizon,
                     t_step=t_step, penalty=penalty)
    simulator = build_simulator(model, params, t_step=t_step)

    # Set initial state for MPC
    mpc.x0 = simulator.x0
    mpc.set_initial_guess()

    n_steps = int(max_time / t_step)

    from tqdm import tqdm
    pbar = tqdm(total=n_steps, desc="MPC running", unit="step")
    for k in range(n_steps):
        u0 = mpc.make_step(simulator.x0)
        simulator.make_step(u0)
        pbar.update(1)

        # Early exit once controller has shut off for several consecutive steps
        active_val = float(u0.flatten()[0])
        if active_val < 0.01:
            # Run a few more steps to confirm it stays off
            settled = True
            for _ in range(min(5, n_steps - k - 1)):
                u0 = mpc.make_step(simulator.x0)
                simulator.make_step(u0)
                pbar.update(1)
                if float(u0.flatten()[0]) > 0.01:
                    settled = False
                    break
            if settled:
                break
    pbar.close()

    return mpc, simulator


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ny = 20
    env_T = 21 + 273.0
    env_RH = 0.65
    C_amb = rh_to_c(env_RH, env_T)

    params = dict(
        ny=ny,
        porosity=0.67,
        rho_s=1100 * 0.39,
        d_pore=4.51e-5,
        tau_thermal=15 * 60,
        bed_height=1e-3,
        bed_length=0.1,
        bed_width=0.1,
        C_amb=C_amb,
        T_env=env_T,
        T0=180 + 273.0,
        RH=env_RH,
        isotherm_file="utils/ev15_uptake.txt",
        kinetics_file="utils/ev15_kinetics.txt",
    )

    experiments = read_experiments()
    max_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

    t_step = 60.0       # 1-minute MPC steps
    n_horizon = 30      # 30-step lookahead (30 min)
    penalty = 1e-5      # tune: larger → earlier cutoff

    mpc, simulator = run_mpc(
        params, max_time,
        t_step=t_step, n_horizon=n_horizon, penalty=penalty,
    )

    # ── Extract results ──
    ts = np.array(simulator.data["_time"]).flatten()
    C_data = np.array(simulator.data["_x", "C"])
    n_data = np.array(simulator.data["_x", "n"])
    T_data = np.array(simulator.data["_x", "T"]).flatten()
    active_data = np.array(simulator.data["_u", "active"]).flatten()

    # Total adsorbed moles
    dy = params["bed_height"] / (ny - 1)
    w = np.ones(ny); w[0] *= 0.5; w[-1] *= 0.5
    dV = w * dy * params["bed_length"] * params["bed_width"]
    sorbent_mass = dV * (1 - params["porosity"]) * params["rho_s"]
    total_moles = np.sum(n_data * sorbent_mass, axis=1)

    # Find termination time
    off_idx = np.where(active_data < 0.01)[0]
    if len(off_idx) > 0:
        t_stop = ts[off_idx[0]]
        print(f"MPC terminated cycle at t = {t_stop:.0f} s  ({t_stop/3600:.2f} h)")
        print(f"Total adsorbed at termination: {total_moles[off_idx[0]]:.6f} mol")
    else:
        print("MPC did not terminate within the simulation window.")

    print(f"Final total adsorbed: {total_moles[-1]:.6f} mol")

    # ── Plot 1: adsorbed moles + active signal ──
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    for i, exp in enumerate(experiments):
        ax1.plot(exp["ElapsedSeconds"].values / 3600,
                 exp["mol_ads"].values, label=f"Exp {i+1}", alpha=0.7)
    ax1.plot(ts / 3600, total_moles, "k-", label="MPC model")
    if len(off_idx) > 0:
        ax1.axvline(t_stop / 3600, color="red", ls="--", label="Cycle end")
    ax1.set_ylabel("Adsorbed [mol]")
    ax1.set_title("MPC-controlled adsorption cycle")
    ax1.legend()

    ax2.step(ts / 3600, active_data, where="post", color="tab:green")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("active")
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title("Control input (active)")
    plt.tight_layout()

    # ── Plot 2: adsorption rate ──
    dt_sample = 120.0
    t_sample = np.arange(0, ts[-1], dt_sample)
    moles_sampled = np.interp(t_sample, ts, total_moles)
    rate = np.diff(moles_sampled) / dt_sample
    t_rate = (t_sample[:-1] + t_sample[1:]) / 2

    fig2, ax3 = plt.subplots()
    ax3.plot(t_rate / 3600, rate)
    if len(off_idx) > 0:
        ax3.axvline(t_stop / 3600, color="red", ls="--", label="Cycle end")
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("Rate [mol/s]")
    ax3.set_title("Adsorption rate")
    ax3.legend()
    plt.tight_layout()

    # ── Plot 3: temperature ──
    fig3, ax4 = plt.subplots()
    ax4.plot(ts / 3600, T_data - 273)
    ax4.set_xlabel("Time [h]")
    ax4.set_ylabel("Temperature [°C]")
    ax4.set_title("Bed Temperature")
    plt.tight_layout()

    plt.show()
