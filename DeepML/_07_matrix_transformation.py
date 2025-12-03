import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here

    determinant_T = torch.linalg.det(T_t)
    determinant_S = torch.linalg.det(S_t)

    if torch.isclose(determinant_T, torch.tensor(0.0)):
        return torch.tensor(-1.0)

    if torch.isclose(determinant_S, torch.tensor(0.0)):
        return torch.tensor(-1.0)

    T_inv = torch.inverse(T_t)

    TA = T_inv.matmul(A_t)
    TAS = TA.matmul(S_t)

    return TAS

if __name__ == '__main__':
    res = transform_matrix(torch.tensor([[1, 2], [3, 4]], dtype=torch.float), torch.eye(2), torch.eye(2))
    print(res.detach().numpy().tolist())

    res = transform_matrix(torch.tensor([[1, 2], [3, 4]], dtype=torch.float),
                           torch.tensor([[2, 0], [0, 3]], dtype=torch.float), torch.eye(2))
    print(res.detach().numpy().tolist())

    res = transform_matrix(torch.tensor([[1, 2], [3, 4]], dtype=torch.float), torch.eye(2),
                           torch.tensor([[1, 0], [0, 0]], dtype=torch.float))
    print(res.detach().numpy().tolist())
    pass
