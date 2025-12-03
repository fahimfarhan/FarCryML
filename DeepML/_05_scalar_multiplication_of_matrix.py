import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    M_mxn = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    M_mxn = M_mxn.multiply(scalar)
    return M_mxn
