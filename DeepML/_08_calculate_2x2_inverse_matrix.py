import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute inverse of a 2Ã2 matrix using PyTorch.
    Input can be Python list, NumPy array, or torch Tensor.
    Returns a 2Ã2 tensor or None if the matrix is singular.
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    det_m = torch.linalg.det(m)
    if torch.isclose(det_m, torch.tensor(0.0)):
        return None

    m_inv = torch.inverse(m)
    return m_inv
