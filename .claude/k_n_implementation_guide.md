# Implementation Guide: Loading-Dependent LDF Kinetics $k(n)$

## Context

The AWH adsorption model (Approach 1 from `Modeling_Adsorption_in_a_finned_AWE_device.pdf`) currently uses a constant LDF rate constant $k = 15 D_\mu / r_s^2$. DVS analysis of the EV-15 sorbent shows that $k$ depends on loading due to two mechanisms: (1) a deliquescence barrier at low loading where dry LiCl must nucleate a liquid phase before fast uptake begins, and (2) progressive pore flooding at higher loading as the dissolved LiCl solution fills the silica pore network. This guide describes how to replace the constant $k$ with a loading-dependent $k(n)$.

## Fitted Model

Replace the constant-$k$ LDF equation:

```
dn/dt = k * (n_eq(C_amb) - n)
```

with:

```
dn/dt = k(n) * (n_eq(C_amb) - n)

k(n) = k0 * exp(-alpha * n) * sigmoid(n)

sigmoid(n) = 1 / (1 + exp(-gamma * (n - nc)))

k0    = 0.001095  [1/s]
alpha = 2.068     [dimensionless, per g/g loading]
gamma = 12.41     [dimensionless, transition sharpness]
nc    = 0.303     [g/g, transition loading]
```

The functional form is an exponential decay (pore filling) multiplied by a sigmoid ramp-up (deliquescence barrier). The sigmoid suppresses $k$ at low loading (dry LiCl, slow solid-state hydration), transitions through $n_c = 0.30$ g/g, and saturates to ~1 for $n > 0.5$ g/g where the exponential decay dominates.

### Fitting Procedure

The initial ~44 minutes of the RH=10% DVS step were trimmed (no measurable mass change — instrument/nucleation lag, not representative of diffusion kinetics). Parameters $\alpha$, $\gamma$, and $n_c$ were then fitted from DVS data (EV-15, 25°C, adsorption steps RH 10–80%) using an integral method minimizing weighted NRMSE across all steps simultaneously. The DVS-fitted $k_0$ was then scaled by 1/10 to match device-scale uptake curves (Exp 1/2/3), accounting for differences between DVS thin-sample geometry and the packed foam bed in the device.

### k(n) Profile

| $n$ [g/g] | $k$ [1/s]  | sigmoid | regime                     |
|-----------|-----------|---------|----------------------------|
| 0.00      | 0.000025  | 0.023   | dry, deliquescence barrier |
| 0.05      | 0.000041  | 0.042   | early hydration            |
| 0.10      | 0.000066  | 0.075   | hydrate formation          |
| 0.15      | 0.000105  | 0.130   | transitioning              |
| 0.20      | 0.000158  | 0.218   | transitioning              |
| 0.30      | 0.000289  | 0.491   | liquid phase forming       |
| 0.40      | 0.000368  | 0.769   | pore filling dominates     |
| 0.50      | 0.000358  | 0.920   | pore filling               |
| 0.70      | 0.000256  | 0.993   | pore filling               |
| 1.00      | 0.000138  | 1.000   | pore filling               |
| 1.50      | 0.000049  | 1.000   | pore filling               |
| 1.90      | 0.000022  | 1.000   | pore filling               |

### DVS Fit Quality (NRMSE per step)

| RH%  | NRMSE |
|------|-------|
| 10   | 3.7%  |
| 20   | 2.5%  |
| 30   | 2.0%  |
| 40   | 1.9%  |
| 50   | 4.1%  |
| 60   | 2.5%  |
| 70   | 3.2%  |
| 80   | 3.3%  |

## Implementation in JAX/Diffrax

### 1. Define the kinetic parameters

```python
import jax.numpy as jnp
from jax.nn import sigmoid

# Loading-dependent LDF parameters (EV-15, 25°C, device-scaled)
K0 = 0.001095    # [1/s] prefactor (DVS value / 10)
ALPHA = 2.068    # [dimensionless] pore-filling decay constant
GAMMA = 12.41    # [dimensionless] deliquescence transition sharpness
NC = 0.303       # [g/g] deliquescence transition loading

def k_ldf(n: jnp.ndarray) -> jnp.ndarray:
    """Loading-dependent LDF rate constant [1/s].
    
    k(n) = k0 * exp(-alpha * n) * sigmoid(gamma * (n - nc))
    
    Two mechanisms:
      - sigmoid: suppresses k at low loading (dry LiCl deliquescence barrier)
      - exp decay: k decreases as pores flood with LiCl solution
    """
    return K0 * jnp.exp(-ALPHA * n) * sigmoid(GAMMA * (n - NC))
```

#### Alternative: KAN literature parameters

The same sigmoid × exp functional form fitted to a dynamic KAN model from literature (different LiCl-silica composite). Higher peak $k$, steeper pore-filling decay, earlier deliquescence transition:

```python
# Alternative: KAN literature fit (intrinsic sorbent values, not device-scaled)
K0 = 0.01986     # [1/s] intrinsic (divide by 10 for device scale → 0.001986)
ALPHA = 2.982    # [dimensionless] steeper pore-filling decay
GAMMA = 17.91    # [dimensionless] sharper deliquescence transition
NC = 0.224       # [g/g] earlier transition loading
```

Parameter comparison (intrinsic / unscaled values):

| Parameter     | DVS (EV-15)  | KAN literature |
|---------------|-------------|----------------|
| $k_0$ [1/s]   | 0.01095     | 0.01986        |
| $\alpha$      | 2.068       | 2.982          |
| $\gamma$      | 12.41       | 17.91          |
| $n_c$ [g/g]   | 0.303       | 0.224          |
| peak $k$ [1/s]| 0.00368     | 0.00633        |
| peak at $n$   | ~0.40       | ~0.35          |

#### Alternative: KAN literature curve (interpolation)

Direct interpolation of the KAN curve rather than fitting an analytical form. More faithful to the original data, no fitting error:

```python
import jax.numpy as jnp

# Digitized from KAN dynamic model (red line)
# Intrinsic values — divide by 10 for device scale
_KAN_N = jnp.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80,
    0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50
])
_KAN_K = jnp.array([
    0.0003, 0.0008, 0.0018, 0.0032, 0.0043, 0.0050, 0.0055, 0.0057,
    0.0055, 0.0052, 0.0048, 0.0043, 0.0038, 0.0032, 0.0027, 0.0019,
    0.0013, 0.0010, 0.0008, 0.0006, 0.0004, 0.0003, 0.0002
])

DEVICE_SCALE = 0.1  # DVS-to-device correction factor

def k_ldf_kan(n: jnp.ndarray) -> jnp.ndarray:
    """KAN literature k(n) via linear interpolation [1/s].
    
    Clamps to boundary values outside [0, 1.5] g/g.
    """
    return jnp.interp(n, _KAN_N, _KAN_K) * DEVICE_SCALE
```

Note: `jnp.interp` is differentiable (piecewise linear) so it works with JAX autodiff if needed for sensitivity analysis, but the derivatives will be piecewise constant.

### 2. Modify the sorbent ODE

The sorbent kinetics equation:

```
dn/dt = k * (n_eq(C_amb) - n)
```

The **only change** is in `k`. Replace:

```python
# OLD: constant k
k = 15 * D_mu / r_s**2  # scalar
dndt = k * (n_eq - n)

# NEW: loading-dependent k
k = k_ldf(n)  # array, same shape as n
dndt = k * (n_eq - n)
```

### 3. Impact on the ODE RHS function

In your Diffrax ODE function, `n` is the state at each axial node along `x`. The change is local — `k_ldf(n)` is evaluated pointwise:

```python
def rhs(t, n, args):
    # Isotherm evaluated at ambient concentration
    n_eq = isotherm(C_amb)
    
    # LDF kinetics — THIS IS THE ONLY CHANGE
    k = k_ldf(n)
    dndt = k * (n_eq - n)
    
    return dndt
```

### 4. Timescale considerations for the solver

With $k(n)$, the LDF timescale varies:

- At $n = 0$ g/g (dry): $k \approx 2.5 \times 10^{-5}$ 1/s → slow but not frozen
- At $n \approx 0.4$ g/g (peak $k$): $k \approx 3.7 \times 10^{-4}$ 1/s → fastest regime
- At $n = 1.9$ g/g: $k \approx 2.2 \times 10^{-5}$ 1/s → slow (pores flooded)

The system is slowest at the extremes and fastest around $n \approx 0.4$ g/g. No changes to the solver choice (implicit RK via Diffrax) should be necessary.

### 5. For the MPC controller

The lumped form for a single fin element:

```python
def adsorption_ode(t, n, RH_amb):
    """Lumped adsorption ODE for MPC.
    
    Single state: n (loading, g/g)
    Input: RH_amb (ambient relative humidity, fractional)
    """
    n_eq = isotherm(RH_amb)
    k = k_ldf(n)
    return k * (n_eq - n)
```

This is a single scalar ODE per fin element — extremely cheap for receding-horizon evaluation.

### 6. Unit conversions

The model uses loading in g/g (grams water per gram dry sorbent). If your existing code uses mol/kg:

```python
MW_water = 0.018  # kg/mol
# Convert: n [g/g] = n [mol/kg] * MW_water
# Both alpha and nc scale accordingly:
ALPHA_MOL_KG = ALPHA * MW_water  # = 2.068 * 0.018 = 0.0372 per mol/kg
NC_MOL_KG = NC / MW_water        # = 0.303 / 0.018 = 16.83 mol/kg
# gamma is dimensionless and scales with nc, so use:
# sigmoid(gamma * (n_mol_kg * MW_water - nc))  OR  sigmoid(gamma * MW_water * (n_mol_kg - nc/MW_water))
```

## Files

- `ev15_ads_swing.csv` — raw DVS data
- `ev15_k_n_analysis.png` — 4-panel analysis figure
- This guide

## Summary of Changes

1. Add `K0 = 0.001095`, `ALPHA = 2.068`, `GAMMA = 12.41`, `NC = 0.303` as model constants
2. Replace `k = 15 * D_mu / r_s**2` with `k = k_ldf(n)` in the LDF rate equation
3. Everything else (BCs, isotherm, solver) stays the same
