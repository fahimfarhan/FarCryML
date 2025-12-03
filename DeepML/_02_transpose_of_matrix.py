import torch
import numpy as np

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t: torch.Tensor = torch.as_tensor(a)
    # Your implementation here
    # it looks like transpose is a 2d matrix thing. if there are multiple dimensions, eg 3d xyz, then we have 3C2 = 3
    # ways to do it. xy (0,1), xz(0,2), and yz(1,2)
    # if 4D tensor, then 4C2 = 12 ways to do transpose. just pick a surface, and transpose the surface.
    # the rest of the dimensions remain the same!
    a_transpose = torch.transpose(a_t, 0, 1)
    return a_transpose

def failed_transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    A: np.ndarray = np.ndarray(a)
    # for numpy, input has dimension axes (0,1,2)
    # suppose we want to swap/transpose on the yz plane. then axes = (0,2,1)

    # in this case, we want xy -> yz. so pass (1,0)
    A_tr = np.transpose(A, (1, 0))
    return A_tr.tolist()
