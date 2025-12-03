

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a[0]) != len(b):
        return -1
    c: list[int|float] = []
    for i in range(0, len(a)):
        x = 0
        for j in range(0, len(b)):
            x += (a[i][j] * b[j])
        c.append(x)
    return c

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    b : list[list[int|float]] = []
    r = len(a)
    c = len(a[0])

    # init new rows of size = c
    for i in range(0, c):
        b.append([])

    for i in range(0, r):
        for j in range(0, c):
            b[j].append(a[i][j])

    return b

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    reshaped_matrix: list[list[int | float]] = []
    r = new_shape[0]
    c = new_shape[1]

    # init new rows of size = c
    for i in range(0, c):
        reshaped_matrix.append([])

    for i in range(0, r):
        for j in range(0, c):
            reshaped_matrix[j].append(a[i][j])

	return reshaped_matrix

def main():
    pass

if __name__ == '__main__':
    main()