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

# Crystal LiCl@Ev15-35 k(n) from LDF fit, mapped to crystal isotherm [1/s]
_CRYSTAL_N = jnp.array([
    0.3083, 0.5466, 0.6878, 0.8393, 1.0092,
])
_CRYSTAL_K = jnp.array([
    0.000139, 0.000257, 0.000267, 0.000286, 0.000242,
])
# k vs pure crystal loading (midpoint of each RH step)
q_crystal = jnp.array([0.0692, 0.2303, 0.4002, 0.5194, 0.5878, 0.6509, \
             0.7238, 0.7918, 0.8704, 0.9647, 1.0560, 1.1543, \
             1.2719, 1.4336, 1.6468, 1.9515])

k_values = jnp.array([0.0008236, 0.0075672, 0.0092382, 0.0078975, 0.0071357, \
            0.0069457, 0.0057886, 0.0049217, 0.0042146, 0.0026883, \
            0.0020750, 0.0019476, 0.0015934, 0.0011755, 0.0007616, \
            0.0005702])

# DVS-extracted k(n) curve [1/s] (material basis, g H2O / g EV-15, adsorption)
_DVS_N = jnp.array([
    0.052, 0.223, 0.408, 0.530, 0.590, 0.644, 0.731, 0.796,
    0.878, 0.967, 1.056, 1.151, 1.263, 1.423, 1.650, 1.926
])
_DVS_K = jnp.array([
    0.0016, 0.0040, 0.0045, 0.0039, 0.0036, 0.0033, 0.0027, 0.0023,
    0.0019, 0.0015, 0.0010, 0.0007, 0.0007, 0.0005, 0.0004, 0.0003
])

# Convert n breakpoints from g/g to mol/kg
_KAN_N_MOL_KG = _KAN_N / WATER_MOLAR_MASS
_CRYSTAL_N_KG = _CRYSTAL_N / WATER_MOLAR_MASS
q_crystal_KG = q_crystal / WATER_MOLAR_MASS
_DVS_N_KG = _DVS_N / WATER_MOLAR_MASS

DEVICE_SCALE = .1  # DVS-to-device correction factor


class SorbentProperties(Module):
    particle_density: Float
    isotherm: Module

    def __init__(self, particle_density, isotherm):
        self.particle_density = particle_density
        self.isotherm = isotherm

    def k_ldf(self, n):
        """KAN literature k(n) via linear interpolation [1/s], device-scaled."""
        return jnp.interp(n, _DVS_N_KG, _DVS_K) * .1# * 8 /15 * 25**2 / 90**2
    def __call__(self, concentration):
        return self.isotherm(concentration)


