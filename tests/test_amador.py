import numpy as np
import torch

from amador_transform import AmadorTransform


def test_project_hexasicortorgonal_enforces_orthogonality():
    """Valida el Axioma IV con una prueba matemática comprobable.

    La memoria antigua se representa como un vector m. Si la proyección es
    correcta, el tensor nuevo proyectado debe quedar completamente ortogonal
    a la memoria antigua, es decir, m · x_proyectado ≈ 0.
    """
    memoria_antigua = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    nuevos_pesos = torch.tensor([4.0, -1.0, 2.0], dtype=torch.float64)

    proyectado = AmadorTransform().project_hexasicortorgonal(nuevos_pesos, memoria_antigua)
    producto_punto = torch.dot(proyectado.reshape(-1), memoria_antigua)

    assert torch.abs(producto_punto).item() < 1e-7
