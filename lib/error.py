'''
byjus.com, Singular Matrix.
[online] Available at: Singular Matrix https://byjus.com/maths/singular-matrix/:
    [0]
'''

import numpy as np


# multilinear regression is not solvable if
class MultilinearSingularError(ValueError):
    def __init__(self, X: np.ndarray) -> None:
        self.error_message = "Multilinear equation not found because X array is singular.\n" \
                             "X = \n{0}".format(X)
        super().__init__(self.error_message)