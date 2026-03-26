"""
1D sorbent bed model (y-direction diffusion + LDF sorption) translated from
diffrax_based/awe_ads.py to do-mpc / CasADi.

Constant concentration (Dirichlet) BC at top, zero-flux (Neumann) at bottom.
Lumped bed temperature relaxing toward ambient.
"""

import do_mpc
import casadi as cas
import numpy as np
import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants & helpers
# ──────────────────────────────────────────────────────────────────────────────
IDEAL_GAS_CONST = 8.314       # J/(mol·K)
WATER_MOLAR_MASS = 0.018      # kg/mol


def psat_water(T):
    return PropsSI("P", "T", float(T), "Q", 0, "Water")


def rh_to_c(rh, T):
    return rh * psat_water(T) / (IDEAL_GAS_CONST * T)


def vapor_diffusivity(T):
    """D_v(T) = D_ref * (T/T_ref)^1.81"""
    return 2.42e-5 * (T / 293.0) ** 1.81


def knudsen_diffusivity(T, d_pore):
    return (d_pore / 3) * cas.sqrt(8 * IDEAL_GAS_CONST * T / (np.pi * WATER_MOLAR_MASS))


def bed_diffusivity(T, porosity, d_pore):
    D_v = vapor_diffusivity(T)
    D_k = knudsen_diffusivity(T, d_pore)
    return porosity ** (3 / 2) * (1 / D_v + 1 / D_k) ** (-1)


# ──────────────────────────────────────────────────────────────────────────────
# Lookup-table loaders  →  CasADi interpolants
# ──────────────────────────────────────────────────────────────────────────────
def _pad_for_clamp(C_vals, y_vals):
    """Pad lookup table with boundary clamp values to mimic jnp.interp behavior.

    CasADi's interpolant extrapolates linearly outside the grid, but jnp.interp
    clamps to the first/last value.  Adding sentinel points far outside the
    physical range with the same boundary values achieves clamping.
    """
    C_lo = C_vals[0] - 1e6
    C_hi = C_vals[-1] + 1e6
    C_vals = [C_lo] + list(C_vals) + [C_hi]
    y_vals = [y_vals[0]] + list(y_vals) + [y_vals[-1]]
    return C_vals, y_vals


def _load_isotherm(path, env_T, env_RH):
    """Return a CasADi interpolant  n_eq(C)  [mol/kg]."""
    data = np.loadtxt(path)
    C_vals = np.array([rh_to_c(rh, env_T) for rh in data[:, 0]])
    n_eq_vals = data[:, 1] / WATER_MOLAR_MASS          # g/g → mol/kg

    # CasADi interpolant needs strictly increasing grid
    order = np.argsort(C_vals)
    C_vals = C_vals[order].tolist()
    n_eq_vals = n_eq_vals[order].tolist()

    C_vals, n_eq_vals = _pad_for_clamp(C_vals, n_eq_vals)
    return cas.interpolant("isotherm", "linear", [C_vals], n_eq_vals)


def _load_k_sorb(path, env_T, env_RH):
    """Return a CasADi interpolant  k_sorb(C)  [1/s]."""
    data = np.loadtxt(path)
    # column 0 is RH values passed raw to rh_to_c (matching diffrax version)
    C_vals = np.array([rh_to_c(rh, env_T) for rh in data[:, 0]])
    k_vals = data[:, 1]

    order = np.argsort(C_vals)
    C_vals = C_vals[order].tolist()
    k_vals = k_vals[order].tolist()

    C_vals, k_vals = _pad_for_clamp(C_vals, k_vals)
    return cas.interpolant("k_sorb", "linear", [C_vals], k_vals)


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────
def build_model(ny=20):
    model = do_mpc.model.Model("continuous", "SX")

    # ── States ──
    C = model.set_variable("_x", "C", shape=(ny, 1))   # gas-phase conc [mol/m³]
    n = model.set_variable("_x", "n", shape=(ny, 1))   # adsorbed amount [mol/kg]
    T = model.set_variable("_x", "T")                   # bed temperature [K]

    # ── Parameters (constant over MPC horizon) ──
    porosity     = model.set_variable("_p", "porosity")
    rho_s        = model.set_variable("_p", "rho_s")
    d_pore       = model.set_variable("_p", "d_pore")
    tau_thermal  = model.set_variable("_p", "tau_thermal")
    bed_height   = model.set_variable("_p", "bed_height")

    # ── Time-varying parameters ──
    C_amb = model.set_variable("_tvp", "C_amb")   # ambient concentration
    T_env = model.set_variable("_tvp", "T_env")   # ambient temperature

    # ── Grid spacing ──
    dy = bed_height / (ny - 1)

    # ── Diffusivity (temperature-dependent) ──
    D_vs = bed_diffusivity(T, porosity, d_pore)

    # ── Sorption: element-wise via map'd interpolant ──
    # Placeholder expressions — the actual interpolants are injected after
    # model.setup() via the simulator's parameter functions, but we need
    # symbolic expressions here.  CasADi interpolants can be called on SX
    # symbols if we map them over the vector dimension.

    # We store the interpolants as module-level objects and call them inside
    # the model definition.  They are created before build_model is called
    # and passed in (see run_simulation below).  To keep build_model
    # self-contained, we accept them as arguments.

    return model, C, n, T, D_vs, dy, porosity, rho_s, C_amb, T_env, tau_thermal


def build_and_setup_model(
    ny, isotherm_interp, k_sorb_interp
):
    model = do_mpc.model.Model("continuous", "SX")

    # ── States ──
    C = model.set_variable("_x", "C", shape=(ny, 1))
    n = model.set_variable("_x", "n", shape=(ny, 1))
    T = model.set_variable("_x", "T")

    # ── Parameters ──
    porosity     = model.set_variable("_p", "porosity")
    rho_s        = model.set_variable("_p", "rho_s")
    d_pore       = model.set_variable("_p", "d_pore")
    tau_thermal  = model.set_variable("_p", "tau_thermal")
    bed_height   = model.set_variable("_p", "bed_height")

    # ── Time-varying parameters ──
    C_amb = model.set_variable("_tvp", "C_amb")
    T_env = model.set_variable("_tvp", "T_env")

    # ── Grid spacing ──
    dy = bed_height / (ny - 1)

    # ── Diffusivity ──
    D_vs = bed_diffusivity(T, porosity, d_pore)

    # ── Sorption RHS (element-wise) ──
    # Map the scalar interpolants over ny elements
    isotherm_map = isotherm_interp.map(ny)
    k_sorb_map   = k_sorb_interp.map(ny)

    n_eq = isotherm_map(C.T).T        # (ny, 1)
    k_s  = k_sorb_map(C.T).T          # (ny, 1)
    dndt = k_s * (n_eq - n)

    # ── 1D diffusion (method of lines) ──
    # Ghost nodes: Dirichlet at top (index 0), Neumann at bottom (index ny-1)
    C_top_ghost = 2 * C_amb - C[0]
    C_bot_ghost = C[ny - 1]

    C_padded = cas.vertcat(C_top_ghost, C, C_bot_ghost)
    d2C_dy2 = (C_padded[2:ny + 2] - 2 * C + C_padded[0:ny]) / dy ** 2

    dCdt = D_vs * d2C_dy2 - (1 - porosity) / porosity * rho_s * dndt

    # ── Temperature ──
    dTdt = (T_env - T) / tau_thermal

    # ── Set RHS ──
    model.set_rhs("C", dCdt)
    model.set_rhs("n", dndt)
    model.set_rhs("T", dTdt)

    model.setup()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Simulator builder
# ──────────────────────────────────────────────────────────────────────────────
def build_simulator(model, params, t_step=1.0):
    """
    params : dict with keys
        porosity, rho_s, d_pore, tau_thermal, bed_height,
        C_amb, T_env, T0, ny
    """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.settings.t_step = t_step
    simulator.settings.abstol = 1e-10
    simulator.settings.reltol = 1e-8

    # ── Fixed parameters ──
    p_template = simulator.get_p_template()
    p_template["porosity"]    = params["porosity"]
    p_template["rho_s"]       = params["rho_s"]
    p_template["d_pore"]      = params["d_pore"]
    p_template["tau_thermal"] = params["tau_thermal"]
    p_template["bed_height"]  = params["bed_height"]

    def p_fun(t_now):
        return p_template
    simulator.set_p_fun(p_fun)

    # ── Time-varying parameters ──
    tvp_template = simulator.get_tvp_template()
    tvp_template["C_amb"] = params["C_amb"]
    tvp_template["T_env"] = params["T_env"]

    def tvp_fun(t_now):
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    # ── Initial conditions ──
    ny = params["ny"]
    simulator.x0["C"] = np.zeros((ny, 1))
    simulator.x0["n"] = np.zeros((ny, 1))
    simulator.x0["T"] = params["T0"]

    return simulator


# ──────────────────────────────────────────────────────────────────────────────
# Run simulation
# ──────────────────────────────────────────────────────────────────────────────
def run_simulation(params, final_time, t_step=1.0):
    """
    params : dict with keys
        porosity, rho_s, d_pore, tau_thermal, bed_height,
        bed_length, bed_width,
        C_amb, T_env, T0, ny,
        isotherm_file, kinetics_file
    """
    ny = params["ny"]

    # Load interpolation tables
    isotherm_interp = _load_isotherm(
        params["isotherm_file"], params["T_env"], params.get("RH", 0.65)
    )
    k_sorb_interp = _load_k_sorb(
        params["kinetics_file"], params["T_env"], params.get("RH", 0.65)
    )

    model = build_and_setup_model(ny, isotherm_interp, k_sorb_interp)
    simulator = build_simulator(model, params, t_step=t_step)

    n_steps = int(final_time / t_step)

    # No control inputs → autonomous system
    u0 = np.zeros((0, 1))

    from tqdm import tqdm
    for _ in tqdm(range(n_steps), desc="Simulating", unit="step"):
        simulator.make_step(u0)

    return simulator


# ──────────────────────────────────────────────────────────────────────────────
# Experiment loader (same as diffrax version)
# ──────────────────────────────────────────────────────────────────────────────
def read_experiments(folder="exp_data/cleaned"):
    folder = Path(folder)
    files = sorted(folder.glob("*_cleaned.csv"))
    experiments = []
    for i, f in enumerate(files):
        if i == 1:
            continue
        df = pd.read_csv(f)
        df["mol_ads"] = df["Ads Weight"] / 420 / 18.01528
        experiments.append(df)
    return experiments


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ny = 20

    env_T = 21 + 273.0   # K
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
    final_time = max(exp["ElapsedSeconds"].iloc[-1] for exp in experiments)

    t_step = 10.0  # seconds per simulator step
    simulator = run_simulation(params, final_time, t_step=t_step)

    # ── Extract results ──
    ts = np.array(simulator.data["_time"]).flatten()
    C_data = np.array(simulator.data["_x", "C"])       # (n_steps, ny)
    n_data = np.array(simulator.data["_x", "n"])       # (n_steps, ny)
    T_data = np.array(simulator.data["_x", "T"]).flatten()

    # ── Compute total adsorbed moles (trapezoidal over y) ──
    dy = params["bed_height"] / (ny - 1)
    w = np.ones(ny)
    w[0] *= 0.5
    w[-1] *= 0.5
    dV = w * dy * params["bed_length"] * params["bed_width"]

    sorbent_mass = dV * (1 - params["porosity"]) * params["rho_s"]
    adsorbed_moles = np.sum(n_data * sorbent_mass, axis=1)
    vapor_moles = np.sum(C_data * params["porosity"] * dV, axis=1)
    total_moles = adsorbed_moles + vapor_moles

    # ── Plot: model vs experiment ──
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for i, exp in enumerate(experiments):
        t_exp = exp["ElapsedSeconds"].values
        mol_exp = exp["mol_ads"].values
        ax1.plot(t_exp / 3600, mol_exp, label=f"Exp {i+1}")

        model_interp = np.interp(t_exp, ts, total_moles)
        pct_error = np.where(mol_exp != 0, (model_interp - mol_exp) / mol_exp * 100, 0)
        ax2.plot(t_exp / 3600, pct_error, label=f"Exp {i+1}")

    ax1.plot(ts / 3600, total_moles, label="Model (do-mpc)")
    ax1.set_ylabel("Ads Moles")
    ax1.set_title("Model vs Experiment")
    ax1.legend()

    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Error (%)")
    ax2.legend()
    plt.tight_layout()

    # ── Plot: temperature ──
    fig2, ax3 = plt.subplots()
    ax3.plot(ts / 3600, T_data - 273, label="Bed temperature")
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("Temperature [°C]")
    ax3.set_title("Lumped Bed Temperature")
    ax3.legend()
    plt.tight_layout()

    # ── Plot: concentration profile snapshots ──
    y_arr = np.linspace(0, params["bed_height"] * 1e3, ny)
    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(10, 5))

    n_snapshots = 6
    snap_idx = np.linspace(0, len(ts) - 1, n_snapshots, dtype=int)
    for idx in snap_idx:
        label = f"t = {ts[idx]/3600:.1f} h"
        ax4.plot(y_arr, C_data[idx], label=label)
        ax5.plot(y_arr, n_data[idx], label=label)

    ax4.set_xlabel("y [mm]")
    ax4.set_ylabel(r"$C_s$ [mol/m³]")
    ax4.set_title("Gas-phase concentration")
    ax4.legend()

    ax5.set_xlabel("y [mm]")
    ax5.set_ylabel(r"$n$ [mol/kg]")
    ax5.set_title("Adsorbed amount")
    ax5.legend()
    plt.tight_layout()

    plt.show()
