# Temperature Adjustment of Intraparticle Mass Transfer Coefficient for LiCl@Hierarchical Silica Composites

## Context and Motivation

This document describes how to adjust a measured intraparticle mass transfer coefficient $k$ [1/s] from a reference temperature $T_\text{ref}$ to an arbitrary target temperature $T$. The material system is a **hierarchical silica sphere** (particle diameter ~20 μm) loaded with hygroscopic **LiCl salt**, designed for atmospheric water harvesting (AWH). The reference data was measured at 25 °C across a range of relative humidities (RH).

The approach is based on the physics described in:

> Chen, C. et al. "Hierarchical Silica Composites for Enhanced Water Adsorption at Low Humidity." *ACS Appl. Mater. Interfaces* **2024**, 16, 40275–40285. DOI: 10.1021/acsami.4c09456

---

## Material Structure

The hierarchical silica (HS-PEG / HS-PEG 2CTAB) has a trimodal pore system:

| Pore type | Size range | Origin | Role |
|-----------|-----------|--------|------|
| Macropores | >50 nm | PEG phase separation | Gas-phase transport highways |
| Secondary mesopores | 10–50 nm | PEG/silica interparticle | Vapor accessibility to interior |
| Primary mesopores | ~4 nm | CTAB templating | LiCl storage sites |

After LiCl impregnation, the micropores and small mesopores are partially or fully filled with salt. The larger mesopores and macropores remain open for vapor transport.

Key material properties (HS-PEG 2CTAB):

- BET surface area: 727 m²/g (bare), 243 m²/g (after LiCl loading)
- Total pore volume: 4.37 cm³/g (bare), reduced after salt loading
- Mesopore volume: 1.44 cm³/g (bare), 0.62 cm³/g (after LiCl loading)
- Bulk porosity: ~81%
- Particle radius: ~10 μm (for 20 μm diameter spheres)

---

## Physics of Intraparticle Transport

### Why the mass transfer coefficient depends on temperature

The measured $k$ [1/s] is a lumped parameter from the linear driving force (LDF) approximation of intraparticle diffusion. For a spherical particle:

$$k = \frac{15\, D_\text{eff}}{R_p^2}$$

where $D_\text{eff}$ is the effective intraparticle diffusivity and $R_p$ is the particle radius. Since $R_p$ is fixed, the temperature dependence of $k$ maps directly from $D_\text{eff}(T)$.

### Why the mechanism changes with RH

In a LiCl-loaded silica particle, the rate-limiting transport mechanism for water vapor shifts with relative humidity:

**Low RH (below deliquescence):** LiCl exists as crystalline hydrates (LiCl·nH₂O) inside the small mesopores. Water must diffuse through or around these solid hydrate phases. This is a thermally activated process governed by **Arrhenius kinetics** with strong temperature sensitivity.

**High RH (above deliquescence):** LiCl absorbs enough water to dissolve and form a concentrated salt solution inside the pores. Water transport now occurs via **liquid-phase diffusion** through this solution, governed by the **Stokes-Einstein relation**. The temperature dependence comes from both the thermal energy term and the viscosity of the LiCl solution.

Bulk LiCl deliquesces at ~47% RH, but nanoconfinement in mesopores shifts this to roughly 40–45% RH.

### Diffusion regimes in the open pores

For the macropores and large mesopores that are NOT salt-loaded, gas-phase water vapor transport follows **Knudsen diffusion** (since the mean free path of H₂O in air at ambient conditions is ~65 nm, comparable to or larger than these pore diameters). The Knudsen diffusivity is:

$$D_K = \frac{d_\text{pore}}{3}\sqrt{\frac{8RT}{\pi M}}$$

This scales as $\sqrt{T}$, which is a weak temperature dependence (~6% increase from 25 °C to 60 °C). Since these open pores are not the rate-limiting step (the salt-loaded pores are), this gas-phase contribution does not dominate $k(T)$ for the composite.

---

## Temperature Adjustment Equations

### Regime 1: Arrhenius (hydrate regime, RH < RH_deliq)

$$k(T) = k(T_\text{ref}) \cdot \exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_\text{ref}}\right)\right]$$

where:
- $E_a$ = activation energy for water transport through LiCl hydrate [J/mol]
- $R$ = 8.314 J/(mol·K)
- $T$, $T_\text{ref}$ in Kelvin

**Typical values:** $E_a$ = 30,000–50,000 J/mol for water in salt hydrate systems. A starting estimate of **40,000 J/mol** is recommended.

**Effect:** Going from 25 °C → 60 °C with $E_a$ = 40 kJ/mol gives a factor of ~2.5× increase in $k$. Going to 80 °C gives ~5.5×.

### Regime 2: Stokes-Einstein (liquid regime, RH > RH_deliq)

$$k(T) = k(T_\text{ref}) \cdot \frac{T}{T_\text{ref}} \cdot \frac{\mu(T_\text{ref})}{\mu(T)}$$

where $\mu(T)$ is the dynamic viscosity of concentrated LiCl solution, approximated as:

$$\mu(T) = A \cdot \exp\!\left(\frac{B}{T}\right)$$

**Typical parameters for ~30–40 wt% LiCl solution:**
- $A$ = 1.0 × 10⁻⁶ Pa·s
- $B$ = 2500 K
- $\mu$(25 °C) ≈ 3–5 mPa·s

**Effect:** Weaker temperature dependence than the Arrhenius regime. Going from 25 °C → 60 °C gives roughly 1.5–2× increase.

### Blending the two regimes

A smooth sigmoid function transitions between hydrate and liquid regimes:

$$\alpha(\text{RH}) = \frac{1}{1 + \exp\!\left(-\frac{\text{RH} - \text{RH}_\text{deliq}}{w/4}\right)}$$

where:
- $\text{RH}_\text{deliq}$ = deliquescence RH of confined LiCl [%], typically ~42% for nanoconfined LiCl
- $w$ = transition width [%RH], typically ~10%

The blended mass transfer coefficient is:

$$k(T, \text{RH}) = (1 - \alpha)\, k_\text{Arrhenius}(T) + \alpha\, k_\text{liquid}(T)$$

### Knudsen contribution (gas-phase, open pores)

If you need to also adjust gas-phase transport through the open macropores:

$$k_\text{Kn}(T) = k(T_\text{ref}) \cdot \sqrt{\frac{T}{T_\text{ref}}}$$

This is typically not rate-limiting for the composite, but relevant for bare (unloaded) silica.

---

## Reference Data

Measured at $T_\text{ref}$ = 25 °C (298.15 K):

| RH (%) | k (1/s) |
|--------|---------|
| 9.9 | 2.4916 × 10⁻⁴ |
| 19.7 | 4.8835 × 10⁻³ |
| 29.5 | 3.6357 × 10⁻³ |
| 39.4 | 2.6772 × 10⁻³ |
| 49.0 | 2.1666 × 10⁻³ |
| 58.7 | 1.4469 × 10⁻³ |
| 68.6 | 8.5696 × 10⁻⁴ |
| 78.3 | 4.3351 × 10⁻⁴ |

The peak at ~20% RH followed by monotonic decay is characteristic of this material: low RH has limited driving force, while increasing RH leads to progressively more liquid-filled pores that slow vapor transport.

---

## Adjustable Parameters

| Parameter | Symbol | Default | Range | How to calibrate |
|-----------|--------|---------|-------|-----------------|
| Activation energy | $E_a$ | 40,000 J/mol | 30,000–50,000 | Fit uptake curves at 2+ temperatures: $\ln(k_2/k_1) = -E_a/R \cdot (1/T_2 - 1/T_1)$ |
| Deliquescence RH | RH_deliq | 42% | 38–47% | From water vapor isotherm: inflection in uptake curve |
| Blend width | $w$ | 10% RH | 5–15% | Controls sharpness of hydrate→liquid transition |
| Viscosity pre-exponential | $A$ | 1.0 × 10⁻⁶ Pa·s | — | Fit to LiCl solution viscosity data at target concentration |
| Viscosity activation temp | $B$ | 2500 K | 2000–3000 | Fit to LiCl solution viscosity data |

### Sensitivity ranking

1. **$E_a$ dominates** the overall temperature response, especially at low RH. A 10 kJ/mol change in $E_a$ shifts the 60 °C prediction by ~2×.
2. **RH_deliq** determines where the transition occurs. Shifting it by ±5% RH changes which data points feel the strong Arrhenius boost.
3. **Viscosity parameters** matter mainly above 50% RH. Their effect on the high-RH points is moderate (~30% variation).

---

## Implementation Guide

### Python function signature

```python
def k_temperature_adjusted(
    RH: float | np.ndarray,       # Relative humidity [%]
    k_ref: float | np.ndarray,    # k at T_ref [1/s]
    T: float,                     # Target temperature [K]
    T_ref: float = 298.15,        # Reference temperature [K]
    E_a: float = 40_000,          # Activation energy [J/mol]
    RH_deliq: float = 42.0,       # Deliquescence RH [%]
    blend_width: float = 10.0,    # Sigmoid transition width [%RH]
    A_visc: float = 1.0e-6,       # Viscosity pre-exponential [Pa·s]
    B_visc: float = 2500.0,       # Viscosity activation temperature [K]
) -> float | np.ndarray:
    """Returns temperature-adjusted k [1/s]."""
```

### Step-by-step algorithm

```
1. Compute Arrhenius-adjusted k:
     k_hyd = k_ref * exp(-E_a / R * (1/T - 1/T_ref))

2. Compute liquid-adjusted k:
     mu_ref = A_visc * exp(B_visc / T_ref)
     mu_T   = A_visc * exp(B_visc / T)
     k_liq  = k_ref * (T / T_ref) * (mu_ref / mu_T)

3. Compute blending weight:
     alpha = 1 / (1 + exp(-(RH - RH_deliq) / (blend_width / 4)))

4. Blend:
     k_adjusted = (1 - alpha) * k_hyd + alpha * k_liq
```

### Integration into a column model

If you are using a packed bed column model with a dimensionless framework, the mass transfer coefficient enters through the number of transfer units (NTU) or a Damköhler-like number. For example:

$$\text{Da}_\text{MT} = \frac{k \cdot L}{u}$$

where $L$ is bed length and $u$ is superficial velocity. Making $k$ temperature-dependent means $\text{Da}_\text{MT}$ also becomes temperature-dependent. If the column is non-isothermal (e.g., during adsorption with heat release of 80–90 kJ/mol), couple the energy balance to update $k$ at each spatial node or timestep.

### For non-isothermal simulations

The heat of adsorption for this material is 80–90 kJ/mol (from TGA/DSC microcalorimetry). During adsorption, the local temperature rises, which transiently changes $D_\text{eff}$ and hence $k$. The temperature rise can be estimated from:

$$\Delta T \approx \frac{\Delta H_\text{ads} \cdot \Delta q}{\rho_b \cdot c_p}$$

where $\Delta q$ is the change in loading, $\rho_b$ is bulk density, and $c_p$ is heat capacity. For a rapid adsorption event, $\Delta T$ can reach 10–30 K locally, which is enough to meaningfully affect the kinetics through the Arrhenius term.

---

## Validation Approach

If you acquire kinetic data at a second temperature (e.g., 40 °C or 60 °C), you can validate and calibrate the model:

1. **Extract $E_a$ directly** from paired measurements at two temperatures at the SAME RH in the hydrate regime (RH < 40%):

$$E_a = -R \cdot \frac{\ln(k_2 / k_1)}{1/T_2 - 1/T_1}$$

2. **Check the liquid regime** by comparing predicted vs. measured $k$ at high RH (>50%) at the new temperature. If the prediction is off, adjust the viscosity parameters.

3. **Refine RH_deliq** by finding the RH at which the temperature sensitivity of $k$ transitions from strong (Arrhenius) to moderate (Stokes-Einstein).

---

## Physical Justification Summary

| RH regime | Dominant pore state | Transport mechanism | T dependence | Scaling |
|-----------|-------------------|---------------------|-------------|---------|
| <~40% RH | LiCl·nH₂O hydrate in small mesopores | Solid-state / surface diffusion | Strong | Arrhenius, exp(-E_a/RT) |
| ~40–50% RH | Transition: hydrate → solution | Mixed | Intermediate | Sigmoid blend |
| >~50% RH | Concentrated LiCl solution in pores | Liquid-phase diffusion | Moderate | T/μ(T) |
| Open macropores (all RH) | Gas phase (not rate-limiting) | Knudsen diffusion | Weak | √T |

The key insight from the Chen et al. paper: bare silica and LiCl-loaded silica have the same heat of adsorption (80–90 kJ/mol), suggesting similar H-bond interactions. But the LiCl-loaded material has much slower kinetics. This confirms that the kinetic bottleneck is mass transport through the salt phase, not the adsorption reaction itself — which is why the temperature dependence of $k$ is governed by diffusion physics (Arrhenius / Stokes-Einstein) rather than reaction kinetics.
