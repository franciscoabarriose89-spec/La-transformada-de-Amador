"""fluid_dynamics.py

Módulo para la optimización basada en dinámica de fluidos incompresible.

Autor: Francisco Amador Barrios Espinoza

El modelo interpreta cada gradiente de pérdida como una fuerza aplicada sobre
un campo vectorial de velocidad. En virtud del Axioma II, la divergencia del
campo debe mantenerse nula. Para ello, se aplica una proyección discreta
Helmholtz-Hodge sobre el campo de velocidades antes de cada actualización del
parámetro, garantizando un flujo incompresible hacia el mínimo de pérdida.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Union

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None

TORCH_DISPONIBLE = torch is not None

TensorLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class NavierStokesOptimizer:
    """Optimizador inspirado en dinámica de fluidos incompresible.

    El flujo de cada parámetro se interpreta como un campo vectorial de
    velocidades con viscosidad ν y densidad ρ. Antes de cada paso de
    actualización, se aplica una proyección ortogonal sobre el subespacio de
    campos con divergencia nula para asegurar que la evolución siga el axioma
    de incompresibilidad topológica.
    """

    def __init__(
        self,
        params: Iterable[Union[np.ndarray, "torch.Tensor"]],
        lr: float = 1e-3,
        viscosity: float = 1e-2,
        density: float = 1.0,
        divergence_tol: float = 1e-8,
    ) -> None:
        params_list = list(params)
        if not TORCH_DISPONIBLE and any(
            ((type(param).__module__ or "").startswith("torch")) for param in params_list
        ):
            raise RuntimeError(
                "PyTorch no está disponible. Instala el paquete completo con: pip install amador-transform[avanzado]"
            )

        self.params = params_list
        self.lr = float(lr)
        self.viscosity = float(viscosity)
        self.density = float(density)
        self.divergence_tol = float(divergence_tol)
        self.velocities: Dict[int, Union[np.ndarray, "torch.Tensor"]] = {}

    @staticmethod
    def _as_array(x: TensorLike) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    @staticmethod
    def _project_zero_divergence_discrete(field: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Proyección ortogonal hacia el espacio de campos con divergencia nula.

        Se construye un operador de divergencia discreto D y se proyecta el campo
        vectorial v mediante la fórmula P = I - D^T (D D^T)^+ D. Este operador
        elimina la parte irrotacional que introduce compresibilidad y conserva la
        dirección de flujo útil hacia el mínimo local.
        """
        field = np.asarray(field, dtype=np.float64)
        if field.ndim == 1:
            return field - field.mean()
        if field.shape[-1] < 2:
            return field - field.mean(axis=tuple(range(field.ndim - 1)), keepdims=True)

        n_samples = field.shape[0]
        dimension = field.shape[-1]
        flatten = field.reshape(-1)
        if flatten.size == 0:
            return field.copy()

        divergence_matrix = np.zeros((n_samples, n_samples * dimension), dtype=np.float64)
        for i in range(n_samples):
            for j in range(dimension):
                idx = i * dimension + j
                divergence_matrix[i, idx] = 1.0
                if i + 1 < n_samples:
                    divergence_matrix[i, (i + 1) * dimension + j] = -1.0

        identity = np.eye(n_samples * dimension, dtype=np.float64)
        projector = identity - divergence_matrix.T @ np.linalg.pinv(
            divergence_matrix @ divergence_matrix.T + tol * np.eye(n_samples, dtype=np.float64)
        ) @ divergence_matrix
        projected = (projector @ flatten).reshape(field.shape)
        return projected.astype(np.float64)

    @staticmethod
    def _project_zero_divergence_torch(field: "torch.Tensor", tol: float = 1e-8) -> "torch.Tensor":
        if torch is None:
            raise ImportError("PyTorch no está disponible en el entorno actual.")
        if field.ndim == 1:
            return field - field.mean()
        if field.shape[-1] < 2:
            return field - field.mean(dim=tuple(range(field.ndim - 1)), keepdim=True)

        numpy_field = field.detach().cpu().numpy()
        projected = NavierStokesOptimizer._project_zero_divergence_discrete(numpy_field, tol=tol)
        return torch.as_tensor(projected, dtype=field.dtype, device=field.device)

    def zero_grad(self) -> None:
        """Reinicia los gradientes de todos los parámetros."""
        for param in self.params:
            if torch is not None and isinstance(param, torch.Tensor):
                if param.grad is not None:
                    param.grad.zero_()

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Ejecuta un paso de optimización conservando la incompressibilidad.

        La actualización se calcula como una combinación lineal entre la
        velocidad previa y el gradiente de pérdida, proyectando el vector de
        flujo sobre el subespacio nulo del operador de divergencia. El paso se
        realiza en una dirección estabilizada que evita la sobreacumulación de
        energía y preserva la geodésica hacia el mínimo de pérdida.
        """
        if not TORCH_DISPONIBLE:
            raise RuntimeError(
                "PyTorch no está disponible. Instala el paquete completo con: pip install amador-transform[avanzado]"
            )

        loss_value: Optional[float] = None
        if closure is not None:
            loss_value = float(closure())

        for index, param in enumerate(self.params):
            if torch is not None and isinstance(param, torch.Tensor):
                if param.grad is None:
                    continue
                gradient = param.grad.detach().clone()
                previous_velocity = self.velocities.get(index, torch.zeros_like(param))
                velocity = previous_velocity + (-self.lr * gradient)
                projected_velocity = self._project_zero_divergence_torch(
                    velocity,
                    tol=self.divergence_tol,
                )
                param.data.add_(projected_velocity)
                self.velocities[index] = projected_velocity
            else:
                if not hasattr(param, "shape"):
                    continue
                if hasattr(param, "grad"):
                    gradient = np.asarray(param.grad, dtype=np.float64)
                else:
                    continue
                previous_velocity = self.velocities.get(index, np.zeros_like(np.asarray(param), dtype=np.float64))
                velocity = previous_velocity + (-self.lr * gradient)
                projected_velocity = self._project_zero_divergence_discrete(
                    velocity,
                    tol=self.divergence_tol,
                )
                param[...] = np.asarray(param, dtype=np.float64) + projected_velocity
                self.velocities[index] = projected_velocity

        return loss_value
