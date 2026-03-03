#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# my code for ch 8 is also from ch7 part d and makes more sense from there but this is the same code used to find if the matrix is diagonalyy dominant
import numpy as np

A = np.array([
    [0.582745, 0.48,     0.10,    0,       0      ],
    [0.48,     1.044129, 0.46,    0.10,    0      ],
    [0.10,     0.46,     1.10431, 0.44,    0.10   ],
    [0,        0.10,     0.44,    0.963889,0.42   ],
    [0,        0,        0.10,    0.42,    0.522565]
], dtype=float)

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]

    for i in range(n):
        diagonal = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)

        if diagonal <= row_sum:
            return False

    return True

print("Is the matrix diagonally dominant?", is_diagonally_dominant(A))

