from equinox import Module, field
import jax
from jaxtyping import Float, Array
import jax.numpy as jnp
from pathlib import Path
from typing import Union
import numpy as np
from CoolProp.CoolProp import PropsSI

def psat_water(T: float) -> float:
    return PropsSI("P", "T", T, "Q", 0, "Water")

def psat_water_jax(T):
    """Antoine equation for water saturation pressure [Pa]. JAX-compatible."""
    # Antoine constants (NIST, valid ~255-373 K), gives pressure in bar
    A, B, C = 5.40221, 1838.675, -31.737
    psat_bar = 10.0 ** (A - B / (T + C))
    return psat_bar * 1e5  # bar -> Pa

def rh_to_c(rh, T):
    return rh*psat_water(T)/(IDEAL_GAS_CONST*T)

def c_to_rh_jax(C, T):
    """Convert molar concentration [mol/m³] to fractional RH. JAX-compatible."""
    return C * IDEAL_GAS_CONST * T / psat_water_jax(T)

IDEAL_GAS_CONST = 8.314 #J/molK
WATER_MOLAR_MASS = .018 # kg/mol

class EnvironmentalConditions(Module):
    
    @staticmethod
    def _c_to_k_converter(T: float) -> float:
        if T < 200: return T+273

        return T
    
    T: Float = field(converter=_c_to_k_converter)
    RH: Float

    @property
    def C_amb(self):
        return rh_to_c(self.RH, self.T)

class AirFlow(Module):
    air_gap_height: Float
    flow_speed: Float
    h_m: Float    # mass transfer coefficient [m/s]


class BedProperties(Module):
    sorbent_bed_height: Float
    sorbent_bed_width: Float
    sorbent_bed_length: Float

    pore_diameter: Float
    porosity: Float
    tau_thermal: Float  # lumped thermal time constant [s]
    T0: Float           # initial bed temperature [K]

    def vapor_diffusivity(self, T):
        """D_v(T) = D_ref * (T / T_ref)^1.81, D_ref = 2.42e-5 m²/s at 293 K."""
        return 2.42e-5 * (T / 293.0) ** 1.81

    def knudsen_diffusivity(self, T):
        return (self.pore_diameter / 3) * \
                jnp.sqrt((8*IDEAL_GAS_CONST*T)/(jnp.pi * WATER_MOLAR_MASS))

    def bed_diffusivity(self, T):
        D_v = self.vapor_diffusivity(T)
        D_k = self.knudsen_diffusivity(T)
        return self.porosity**(3/2) * ((1/D_v + 1/D_k) ** (-1))

class Isotherm(Module):
    C: Array
    n_eq: Array

    
    @staticmethod
    def g_per_g_to_mol_per_kg(n_eq):
        return n_eq / WATER_MOLAR_MASS

    @classmethod
    def read_from_file(cls, path: Union['str', Path], env: EnvironmentalConditions) -> Isotherm:
        path = Path(path)

        data = np.loadtxt(path)

        C = rh_to_c(data[:, 0], env.T)
        n_eq = cls.g_per_g_to_mol_per_kg(data[:, 1])

        return cls(C=C, n_eq=n_eq)
    
    def __call__(self, concentration):
        return jnp.interp(concentration, self.C, self.n_eq)

class SorbentProperties(Module):
    particle_radius: Float
    particle_diffusivity: Float
    particle_density: Float
    isotherm: Module

    k_sorb_RH_file: Array   # RH values from file (fractional)
    k_sorb_C_file: Array
    k_sorb_from_file: Array
    T_ref: Float             # reference temperature for k data [K]

    # Temperature adjustment parameters
    E_a: Float               # Arrhenius activation energy [J/mol]
    RH_deliq: Float          # deliquescence RH (fractional)
    blend_width: Float       # sigmoid transition width (fractional RH)
    A_visc: Float            # viscosity pre-exponential [Pa·s]
    B_visc: Float            # viscosity activation temperature [K]

    def __init__(self, particle_radius, particle_diffusivity, particle_density, isotherm, k_sorb_file, env,
                 E_a=40_000.0, RH_deliq=0.42, blend_width=0.10, A_visc=1.0e-6, B_visc=2500.0):
        self.particle_radius = particle_radius
        self.particle_diffusivity = particle_diffusivity
        self.particle_density = particle_density
        self.isotherm = isotherm

        path = Path(k_sorb_file)
        data = np.loadtxt(path)

        self.k_sorb_RH_file = jnp.array(data[:, 0])  # fractional RH
        self.k_sorb_C_file = rh_to_c(data[:,0], env.T)
        self.k_sorb_from_file = data[:, 1]
        self.T_ref = env.T

        self.E_a = E_a
        self.RH_deliq = RH_deliq
        self.blend_width = blend_width
        self.A_visc = A_visc
        self.B_visc = B_visc

    @property
    def k_sorb(self) -> Float:
        return 15 * self.particle_diffusivity / (self.particle_radius**2)

    def k_sorb_C(self, concentration, T=None) -> Float:
        """LDF rate constant. If T is provided, applies temperature adjustment."""
        k_ref = jnp.interp(concentration, self.k_sorb_C_file, self.k_sorb_from_file)
        if T is None:
            return k_ref
        return self._k_temperature_adjusted(k_ref, concentration, T)

    def _k_temperature_adjusted(self, k_ref, concentration, T):
        R = IDEAL_GAS_CONST
        T_ref = self.T_ref

        # Arrhenius regime (hydrate, low RH)
        k_hyd = k_ref * jnp.exp(-self.E_a / R * (1.0 / T - 1.0 / T_ref))

        # Stokes-Einstein regime (liquid, high RH)
        mu_ref = self.A_visc * jnp.exp(self.B_visc / T_ref)
        mu_T = self.A_visc * jnp.exp(self.B_visc / T)
        k_liq = k_ref * (T / T_ref) * (mu_ref / mu_T)

        # Blend based on local RH
        RH = c_to_rh_jax(concentration, T)
        alpha = 1.0 / (1.0 + jnp.exp(-(RH - self.RH_deliq) / (self.blend_width / 4.0)))

        return (1.0 - alpha) * k_hyd + alpha * k_liq

    def __call__(self, concentration):
        return self.isotherm(concentration)


