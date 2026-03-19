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

    k_sorb_C_file: Array
    k_sorb_from_file: Array

    def __init__(self, particle_radius, particle_diffusivity, particle_density, isotherm, k_sorb_file, env):
        self.particle_radius = particle_radius
        self.particle_diffusivity = particle_diffusivity
        self.particle_density = particle_density
        self.isotherm = isotherm

        path = Path(k_sorb_file)
        data = np.loadtxt(path)

        self.k_sorb_C_file = rh_to_c(data[:,0], env.T)
        self.k_sorb_from_file = data[:, 1]
        print(self.k_sorb_C_file)
        print(self.k_sorb_from_file)

    @property
    def k_sorb(self) -> Float:
        return 15 * self.particle_diffusivity / (self.particle_radius**2)

    def k_sorb_C(self, concentration) -> Float:
        return jnp.interp(concentration, self.k_sorb_C_file, self.k_sorb_from_file)

    def __call__(self, concentration):
        return self.isotherm(concentration)


