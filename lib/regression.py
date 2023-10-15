'''
references

Rogers, Simon, and Mark Girolami. (2016). A First Course in Machine Learning, CRC Press LLC.
[online] Available at https://ebookcentral-proquest-com.proxy.library.lincoln.ac.uk/lib/ulinc/reader.action?docID=4718644:
    [0] = page 7
    [2] = page 6
    [3] = pages 9 - 12
    [4] = pages 19 - 25

Petra Bosilj. (September 26, 2023). Introduction to Machine Learning, University of Lincoln:
[online] Available on Blackboard:
    [1] = slide 26 (/30)

Wikipedia. (2022). Vandermonde matrix.
[online] Available at: https://en.wikipedia.org/wiki/Vandermonde_matrix#.
    [5] = (top)

Dr Christos Frantzidis. (October 3, 2023). Model selection & evaluation, University of Lincoln.
[online] Available on Blackboard:
    [6] = slides 17 - 25 (/26)

Wikipedia. (2022). Coefficient of Determination
[online] Available at: https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions:
    [7] = (top)
'''

from icecream import ic

import numpy as np
from typing import Callable

from error import *


# generates a distribution 2d of an equation
def generate_distribution(
    formula: Callable,
    mu     : int | float,
    sigma  : int | float,
    number : int,
    start  : int | float,
    end    : int | float,
    input_v: bool = False,
    shuffle: bool = False
    ) -> np.ndarray:

    rand_Y = np.random.normal(mu, sigma, number)
    if shuffle: np.random.shuffle(rand_Y)
    linspace = np.linspace(start, end, number)

    if not input_v: return formula(linspace) + rand_Y

    return np.array([linspace, formula(linspace) + rand_Y])


# finds the linear regression argument of a line when D = {xi, yi}, i = 1...N [1]
# where y = f(x;w0,w1) = w0 + w1 * x [2]
def simple_linear_regression(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    # there must be as many x and y data points
    assert len(x) == len(y)

    # [3] our loss function is L = 1/N * Σ(w1^2 * xn^2 + 2w1 * xn * w0 - 2w1 * xn * yn)
    # we must find the partial derivative for L with respect to w0

    # we then equate the derivative to zero to find the mininum
    # dL/dw0 = 2w0 + 2w1 * 1/N * Σxn - 2/N Σyn = 0
    # since mean(y) = 1/N Σyn and mean(x) = 1/N * Σxn,
    # 2w0 + 2w1 * 1/N * Σxn - 2/N Σyn = 0 → mininum(w0) = mean(y) - w1 * mean(x)

    y_mean = y.mean()
    x_mean = x.mean()

    # [3] we need to find mininum(w1) to get mininum(w0)
    # so we find the partial derivative of L with respect to w1
    # dL/dw1 = w1 * 2/N * Σxn^2 + 2/N * Σxn(mininum(w0) - yn)
    #        = w1 * 2/N * Σxn^2 + 2/N * Σxn(mean(y) - w1 * mean(x) - yn)
    #        = 2w1 * (1/N * Σxn^2 - mean(x)^2) + 2 * mean(y) * mean(x) - 2 * 1/N Σ(xn * yn)
    #        = 0

    # then we rearange for mininum(w1):
    # mininum(w1) = (1/N * Σ(xn * yn) - mean(x) * mean(y)) / (1/N * Σxn^2 - mean(x)^2)

    # simplifying 1/N * Σxn^2 to mean(x^2)
    #         and 1/N * Σ(xn * yn) to mean(xy),
    # we get the formula: mininum(w1) = (mean(xy) - mean(x) * mean(y)) / (mean(x^2) - mean(x)^2)

    xy_mean = (x * y).mean()
    x2_mean = (x ** 2).mean()

    w1_mininum = (xy_mean - (x_mean * y_mean)) / (x2_mean - (x_mean ** 2))
    w0_mininum = y_mean - w1_mininum * x_mean

    return np.array([w0_mininum, w1_mininum])


def simple_polynomial_regression(y: np.ndarray, x: np.ndarray, M: int) -> np.ndarray:
    # there must be as many x and y data points
    assert len(x) == len(y)
    # there also cant be less parameters than zero and more parameters than data
    assert 0 < M < len(x)

    # [4] we will again use a loss function to determine the line but translated to vector form
    # w = [w0, w1, ..., wM], xn = [1, xn, xn^2, ..., xn^(M-1)]
    # our function is now f(xn; w) = w^T * xn or dot(w, xn)
    # therefor loss L = 1 / 1/N * Σ(yn - w^T * xn)^2.

    # We can use a Vandermonde matrix which is a matrix "with the terms of a geometric
    # progression in each row" [5] to combine all of our xn terms (X) [4].
    # [4] this allows us to simplify our loss function:
    # L = 1/N (t - X * w)^T(t - X * w)

    # [4] like in linear interpolation we want to find the mininum value for our weights.
    # to do this we must find the mininum Loss for by equating dL/dw = 0
    # dL/dw = 2/N * X^T * X * w - 2/N * X^T * t, so
    # X^T * X * w = X^T * t, therefor argmin(w) = (X^T * X)^(-1) * X^T * t

    # Vandermonde matrix
    X = np.matrix([
        [xn ** m for m in range(M)]
        for xn in x
    ])

    # polynomial arguments
    w = np.array((np.linalg.inv(X.T * X) * X.T).dot(y))

    return w


def multilinear_regression(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # there must be as many xn and y data points
    # assert X.shape[1] == len(y)
    # if X is 1-D, force it to be 2-D
    # if len(X) == X.size:
    #     X = X.resize(-1,1)

    # entirely the same as simple regression except we must prefix each
    # level of X with 1 so that it aligns with y = ... + w0x0

    X = np.column_stack((np.ones(len(X)), X))

    W = None
    try:
        W = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
    except np.linalg.LinAlgError:
        raise MultilinearSingularError(X[...,1:])

    return W


# converts the regression array into a linear model
def make_regression_function(A: np.ndarray) -> Callable:
    # clossure function for regression line
    def function(x: int | float | np.ndarray) -> float | np.ndarray:
        if type(x) != np.ndarray:
            return float((np.array([x**m for m in range(len(A))]).dot(A)).sum())
        return np.array([
            (np.array([val**m for m in range(len(A))]).dot(A)).sum()
            for val in x
        ])
    return function


# [6] aka MAE, used to messure the error of a function
def mean_absolute(y: np.ndarray, x: np.ndarray, regression_f: Callable) -> float:
    assert len(y) == len(x)
    N = len(x)
    return (1 / N) * np.abs(y - regression_f(x)).sum()


# [6] aka RMSE, used to messure the error of a function in terms of the dependant variable
def root_mean_squared(y: np.ndarray, x: np.ndarray, regression_f: Callable) -> float:
    assert len(y) == len(x)
    N = len(x)
    return np.sqrt((1 / N) * ((y - regression_f(x)) ** 2).sum())


# [6] [7] aka RMSE, used to messure the error of a function in terms of the dependant variabl
def R_squared(y: np.ndarray, x: np.ndarray, regression_f: Callable) -> float:
    e = y - regression_f(x)
    SSresidual = (e ** 2).sum()
    SStotal = ((y - y.mean()) ** 2).sum()
    return 1 - (SSresidual / SStotal)


# applies an error function
def model_error(y: np.ndarray, x: np.ndarray, regression_f: Callable, loss_f: Callable) -> float:
    assert len(y) == len(x)
    N = len(y)
    return (1 / N) * (loss_f(y, x, regression_f)).sum()


