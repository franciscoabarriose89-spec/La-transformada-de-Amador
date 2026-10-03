# amador-transform

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776AB.svg" />
  <img alt="NumPy" src="https://img.shields.io/badge/numpy-1.24%2B-013243.svg" />
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg" />
</p>

## Autor

Francisco Amador Barrios Espinoza

## Resumen Ejecutivo

`amador-transform` representa un avance científico y tecnológico orientado a resolver uno de los problemas fundamentales de la Inteligencia Artificial contemporánea: el olvido catastrófico. La librería propone una reformulación del aprendizaje como un proceso dinámico sobre colectores continuos, concretamente variedades de Kähler, donde el cálculo discreto se proyecta a un espacio continuo para estabilizar la memoria, conservar la topología y limitar la entropía del sistema.

El núcleo del paradigma es la proyección `Hexasicortorgonal`, que separa ortogonalmente la nueva información del espacio de memoria ya consolidado. En este marco, la memoria antigua no es reemplazada ni borrada por los nuevos pesos: se mantiene en una base estable, mientras que las actualizaciones entran en un complemento ortogonal que evita la interferencia destructiva. La consecuencia directa es la eliminación de la dinámica de solapamiento responsable del olvido catastrófico.

La evolución del optimizador está gobernada por una formulación inspirada en la ecuación de Navier-Stokes, con divergencia cero:

$$
\nabla \cdot v = 0
$$

lo que garantiza flujo incompresible, trayectorias estables y convergencia hacia mínimos de pérdida sin explosión combinatoria ni colapso de la memoria. En términos de información, el sistema se encuentra en una condición de Entropía Cero:

$$
\Delta S = 0
$$

La combinación de estas tres ideas —Variedad de Kähler, flujo incompresible y proyección ortogonal— redefine la optimización como un mecanismo topológico de preservación de memoria absoluta.

## Abstract

`amador-transform` is a topological optimization framework designed to solve catastrophic forgetting in modern artificial intelligence by transferring the discrete computational state into a continuous Kähler manifold and enforcing a null-divergence flow over the optimization landscape. The system reformulates learning dynamics as a geometric and fluidic process in which memory is treated as an invariant manifold rather than as a transient set of weights susceptible to overwrite-induced forgetting.

The central mechanism is the `Hexasicortorgonal projection`, a mathematically grounded orthogonal decomposition that prevents new updates from interfering with previously consolidated memory. In practical terms, new tensors are projected onto the orthogonal complement of the archived memory basis, preserving the historical state while allowing new adaptation to occur in an independent topological direction. Combined with a Navier-Stokes-inspired optimizer, this yields an incompressible flow field with zero divergence, stabilizing convergence and suppressing combinatorial blow-up.

This design enforces the computational isentropy axiom:

$$
\Delta S = 0
$$

preserving memory as a non-vanishing topological invariant and establishing an alternative route for robust continual learning in high-dimensional, non-stationary environments.

## ¿Qué resuelve esta librería?

La Inteligencia Artificial tradicional sufre un fenómeno bien conocido: el olvido catastrófico. Cuando la red aprende una nueva tarea, los pesos asociados a la información anterior se degradan o se sobrescriben. `amador-transform` introduce una solución geométrica y dinámica:

- proyecta el espacio discreto hacia una variedad continua,
- convierte la optimización en un problema de flujo incompresible,
- mantiene memoria consolidada en un subespacio estable,
- evita la interferencia de nuevos pesos sobre la base antigua mediante proyección ortogonal.

Este enfoque transforma la memoria en un invariante topológico, no en un estado frágil de actualización local.

## Instalación

```bash
pip install amador-transform
```

## Quick Start

```python
import numpy as np

from amador_transform import AmadorTransform, KahlerManifold, NavierStokesOptimizer

# 1) Construcción del grafo discreto y proyección a una variedad continua
adjacency = np.array(
    [
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 0.0],
    ],
    dtype=np.float64,
)

manifold = KahlerManifold(dimension=2, bandwidth=1.0)
positions, sdf, laplacian = manifold.project_graph_to_manifold(adjacency)
print("Posiciones:", positions.shape)
print("SDF:", sdf)
print("Laplaciano:", laplacian.shape)

# 2) Optimización con flujo incompresible y divergencia cero
parameters = [np.ones((4, 2), dtype=np.float64)]
optimizer = NavierStokesOptimizer(parameters, lr=1e-2, viscosity=1e-3)

# 3) Proyección Hexasicortorgonal para evitar el olvido catastrófico
memoria_antigua = np.array([1.0, 0.0, 2.0], dtype=np.float64)
nuevos_pesos = np.array([2.5, -1.5, 0.5], dtype=np.float64)
proyectado = AmadorTransform().project_hexasicortorgonal(nuevos_pesos, memoria_antigua)
print("Tensor proyectado:", proyectado)
print("Producto punto con memoria antigua:", np.dot(proyectado, memoria_antigua))
```

## Axiomas del Paradigma Amador

### Axioma I — Isentropía Computacional
Se exige que la variación de entropía del flujo computacional sea nula:

$$
\Delta S = 0
$$

Esto impide la explosión combinatoria y estabiliza la evolución del sistema.

### Axioma II — Incompresibilidad Topológica
La dinámica del optimizador se comporta como un flujo de Navier-Stokes con divergencia cero:

$$
\nabla \cdot v = 0
$$

lo que garantiza que el sistema avance hacia el mínimo sin compresión ni pérdida de estructura.

### Axioma III — Desingularización Ortogonal
La geometría del problema incorpora una proyección que evita atascos en mínimos locales y estabiliza la convergencia.

### Axioma IV — Superposición Hexasicortorgonal
Los nuevos tensores son proyectados fuera del subespacio de memoria consolidada. La memoria absoluta se conserva y la interferencia destructiva desaparece.

## Módulos principales

- `KahlerManifold`: proyecta grafos discretos y topologías finitas a un espacio continuo con SDF.
- `NavierStokesOptimizer`: actualiza parámetros bajo un flujo incompresible y matemáticamente estable.
- `AmadorTransform`: aplica la proyección Hexasicortorgonal para asegurar persistencia de memoria y prevenir olvido catastrófico.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.

## Nota final

`amador-transform` no es una simple librería de optimización. Es una arquitectura conceptual para la estabilidad de sistemas inteligentes de alta dimensión, integrando topología, mecánica de fluidos, geometría compleja y memoria persistente. Su valor principal es la capacidad de proteger la información aprendida frente a la interferencia y el reemplazo estructural, convirtiéndose en una base sólida para la próxima generación de sistemas de inteligencia adaptativa.
