'''REFERENCES AT THE BOTTOM OF THE PAGE'''

import numpy as np


# multilinear regression is not solvable if the matrix X is singular.
# [0] a singular matrix is when the determinent is 0 for a matrix
class MultilinearSingularError(ValueError):
    def __init__(self, X: np.ndarray) -> None:
        self.error_message = "Multilinear equation not found because X array is singular.\n" \
                             "X = \n{0}".format(X)
        super().__init__(self.error_message)

'''
byjus.com, Singular Matrix.
[online] Available at: Singular Matrix https://byjus.com/maths/singular-matrix/:
    [0] = What is Singular Matrix?
'''