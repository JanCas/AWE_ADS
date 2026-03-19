import equinox as eqx
from jaxtyping import Array, Float
from typing import Callable
import jax
import jax.numpy as jnp

# Represents the interval [x0, x_final] discretised into n equally-spaced points.
class SpatialDiscretisation(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    vals: Float[Array, "n"]

    @classmethod
    def discretise_fn(cls, x0: float, x_final: float, n: int, fn: Callable):
        if n < 2:
            raise ValueError("Must discretise [x0, x_final] into at least two points")
        vals = jax.vmap(fn)(jnp.linspace(x0, x_final, n))
        return cls(x0, x_final, vals)

    @property
    def δx(self):
        return (self.x_final - self.x0) / (len(self.vals) - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.x0 != other.x0 or self.x_final != other.x_final:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(self.x0, self.x_final, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)


class SpatialDiscretisation2D(eqx.Module):
    x0: float = eqx.field(static=True)
    x_final: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)
    y_final: float = eqx.field(static=True)
    vals: Float[Array, "ny nx"]

    @property
    def δx(self):
        return (self.x_final - self.x0) / (self.vals.shape[1] - 1)

    @property
    def δy(self):
        return (self.y_final - self.y0) / (self.vals.shape[0] - 1)

    @property
    def trapez_weights(self):
        """Trapezoidal integration weights: edges 1/2, corners 1/4."""
        w = jnp.ones_like(self.vals)
        w = w.at[0, :].multiply(0.5)
        w = w.at[-1, :].multiply(0.5)
        w = w.at[:, 0].multiply(0.5)
        w = w.at[:, -1].multiply(0.5)
        return w * self.δx * self.δy