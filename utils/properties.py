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

def rh_to_c(rh, T):
    return rh*psat_water(T)/(IDEAL_GAS_CONST*T)

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

# Digitized KAN literature k(n) curve [1/s] (intrinsic, in g/g)
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
# Convert n breakpoints from g/g to mol/kg
_KAN_N_MOL_KG = _KAN_N / WATER_MOLAR_MASS

DEVICE_SCALE = 0.1  # DVS-to-device correction factor


class SorbentProperties(Module):
    particle_density: Float
    isotherm: Module

    def __init__(self, particle_density, isotherm):
        self.particle_density = particle_density
        self.isotherm = isotherm

    def k_ldf(self, n):
        """KAN literature k(n) via linear interpolation [1/s], device-scaled."""
        return jnp.interp(n, _KAN_N_MOL_KG, _KAN_K) * DEVICE_SCALE

    def __call__(self, concentration):
        return self.isotherm(concentration)


