"""amador_transform

Biblioteca de optimización y proyección topológica para resolver complejidad
hiperdimensional mediante un mapeo discreto-continuo sobre variedades de Kähler.

Autor: Francisco Amador Barrios Espinoza

La librería implementa el paradigma tecnológico desarrollado por el Investigador
Independiente Francisco Amador Barrios Espinoza, que transfiere el cálculo
algebraico discreto a un colector continuo para alcanzar estabilidad, memoria
absoluta y aprendizaje incompresible en espacios de alta dimensión.
"""

from __future__ import annotations

from .fluid_dynamics import NavierStokesOptimizer
from .operators import AmadorTransform
from .topology import KahlerManifold

__version__ = "1.0.0"
__author__ = "Francisco Amador Barrios Espinoza"
__all__ = [
    "AmadorTransform",
    "KahlerManifold",
    "NavierStokesOptimizer",
    "__author__",
    "__version__",
]
