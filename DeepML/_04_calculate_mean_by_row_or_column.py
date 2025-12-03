import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    A_mxn = torch.as_tensor(matrix, dtype=torch.float)
    if "column" == mode:
        A_n = torch.mean(A_mxn, dim = 0)
        return A_n
    elif "row" == mode:
        A_m = torch.mean(A_mxn, dim = 1)
        return A_m
    else:
        raise Exception(f"Invalid mode {mode}")

def calculate_matrix_meanV1(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    A_mxn: torch.Tensor = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == "column":
        # print("Calculating mean of column mode")
        return calculate_matrix_mean_column(A_mxn)
    elif mode == "row":
        # print("Calculating mean of row mode")
        return calculate_matrix_mean_row(A_mxn)
    else:
        raise ValueError("Invalid mode")
    pass


def calculate_matrix_mean_column(A_mxn: torch.Tensor) -> torch.Tensor:
    m: int = A_mxn.shape[0]
    n: int = A_mxn.shape[1]
    avg_list = []
    for c in range(n): # column
        num = 0.0
        for r in range(m): # row
            num += A_mxn[r, c]
        avg_list.append(num / m)
    A_m = torch.tensor(avg_list, dtype=torch.float)
    return A_m

def calculate_matrix_mean_row(A_mxn: torch.Tensor) -> torch.Tensor:
    m: int = A_mxn.shape[0]
    n: int = A_mxn.shape[1]
    avg_list = []

    for r in range(m):
        num = 0.0
        for c in range(n):
            num += A_mxn[r, c]
        avg_list.append(num/n)

    A_n = torch.tensor(avg_list, dtype=torch.float)
    # A_n_transpose = torch.transpose(A_n, 0, 1)
    # return A_n_transpose
    return A_n
