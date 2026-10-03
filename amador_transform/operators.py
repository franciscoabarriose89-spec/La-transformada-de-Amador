"""operators.py

Implementación principal del transformador de memoria y proyección ortogonal
sobre un espacio de memoria persistente.

Autor: Francisco Amador Barrios Espinoza

El objetivo del operador es proyectar nuevos tensores sobre la memoria
acumulada del sistema, de modo que el aprendizaje se mantenga estable y no se
pierda información antigua. Este procedimiento se formaliza como una
proyección ortogonal sobre el espacio nulo conformado por la memoria previa,
lo que equivale a una descomposición del flujo en una componente retenida y
otra componente descartada.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None

TensorLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class AmadorTransform:
    """Transforma tensores y estados de memoria bajo una proyección ortogonal.

    El método principal project_hexasicortorgonal proyecta un conjunto de tensores
    sobre el espacio generado por la memoria previamente almacenada. El cómputo
    se realiza con proyección matricial de rango bajo, lo que estabiliza la
    información y reduce el riesgo de olvido catastrófico.
    """

    def __init__(self, regularization: float = 1e-8) -> None:
        self.regularization = float(regularization)

    @staticmethod
    def _as_numpy_array(data: Any) -> np.ndarray:
        if torch is not None and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data, dtype=np.float64)

    @staticmethod
    def _reshape_tensor(tensor: Any) -> np.ndarray:
        array = AmadorTransform._as_numpy_array(tensor)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array.reshape(array.shape[0], -1)

    @staticmethod
    def _restore_shape(source: Any, projected: np.ndarray) -> Any:
        if torch is not None and isinstance(source, torch.Tensor):
            original_shape = source.shape
            restored = projected.reshape(original_shape)
            return torch.as_tensor(restored, dtype=source.dtype, device=source.device)
        original_shape = np.asarray(source).shape
        return projected.reshape(original_shape)

    def project_hexasicortorgonal(
        self,
        tensors: TensorLike,
        null_space: Optional[TensorLike],
    ) -> Any:
        """Proyecta tensores al complemento ortogonal del espacio de memoria existente.

        El Axioma IV exige que los nuevos pesos no interfieran con la memoria
        previamente consolidada. Por tanto, la proyección no busca reforzar el
        subespacio antiguo, sino excluirlo: se calcula el complemento ortogonal
        del espacio generado por la memoria previa. Matemáticamente, si M es la
        base de memoria, entonces la proyección del complemento es

            P_⊥ = I - M (M^T M + λI)^-1 M^T

        con λ > 0 como regularización. Así, cualquier tensor nuevo queda
        ortogonal a la base de memoria, evitando el solapamiento que destruye
        la información antigua.

        Parámetros:
            tensors: nuevas observaciones o tensores a proyectar.
            null_space: subespacio de memoria previa, ya sea un vector o una
                colección de vectores de memoria.

        Retorno:
            Un nuevo tensor o conjunto de tensores proyectados en el complemento
            ortogonal de la memoria antigua, preservando la forma original del
            input para interoperabilidad con NumPy o PyTorch.
        """
        if null_space is None:
            return tensors

        memory_basis = self._as_numpy_array(null_space)
        if memory_basis.ndim == 0:
            memory_basis = memory_basis.reshape(1, 1)
        if memory_basis.ndim == 1:
            memory_basis = memory_basis.reshape(1, -1)
        if memory_basis.ndim != 2:
            raise ValueError("El espacio nulo debe ser un vector o una matriz 2D.")

        source_array = self._reshape_tensor(tensors)
        if source_array.shape[1] != memory_basis.shape[-1]:
            raise ValueError(
                "La dimensión interna de los tensores debe coincidir con la dimensión del espacio de memoria."
            )

        memory_dim = memory_basis.shape[-1]
        memory_matrix = memory_basis.T
        gram_matrix = memory_matrix.T @ memory_matrix
        regularized = gram_matrix + self.regularization * np.eye(gram_matrix.shape[0], dtype=np.float64)
        memory_projector = memory_matrix @ np.linalg.pinv(regularized) @ memory_matrix.T
        projector = np.eye(memory_dim, dtype=np.float64) - memory_projector
        projected = source_array @ projector.T

        if torch is not None and isinstance(tensors, torch.Tensor):
            return torch.as_tensor(projected.reshape(tensors.shape), dtype=tensors.dtype, device=tensors.device)

        original_shape = np.asarray(tensors).shape
        return projected.reshape(original_shape)
