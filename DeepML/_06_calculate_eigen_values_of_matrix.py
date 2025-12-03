import torch
import numpy as np

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    eigen_values: torch.Tensor = torch.linalg.eigvals(matrix)
    real_eigen_values = eigen_values.real
    sorted_real_eigen_values, ignore_long_tensor = torch.sort(real_eigen_values)

    return sorted_real_eigen_values

def calculate_eigenvalues_v2(matrix: torch.Tensor) -> torch.Tensor:
    """
        Compute eigenvalues of a 2Ã2 matrix using PyTorch.
        Input: 2Ã2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
        """
    # Your implementation here
    eigen_values: torch.Tensor = torch.linalg.eigvals(matrix)
    mlist = eigen_values.numpy().real
    mlist = np.sort(mlist)
    eigen_values = torch.Tensor(mlist)
    return eigen_values


def calculate_eigenvalues_v1(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2Ã2 matrix using PyTorch.
    Input: 2Ã2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    # Your implementation here
    eigen_values: torch.Tensor = torch.linalg.eigvals(matrix)
    mlist = eigen_values.numpy()
    mlist = np.real(mlist)
    mlist = mlist.tolist()

    mlist = np.sort(mlist)

    eigen_values = torch.Tensor(mlist)
    return eigen_values


if __name__ == '__main__':
    res = calculate_eigenvalues(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    print(res.detach().numpy().tolist())

    res = calculate_eigenvalues(torch.tensor([[0.0,1.0],[1.0,0.0]]))
    print(res.detach().numpy().tolist())

    res = calculate_eigenvalues(torch.tensor([[4.0, 2.0], [1.0, 3.0]]))
    print(res.detach().numpy().tolist())
    pass