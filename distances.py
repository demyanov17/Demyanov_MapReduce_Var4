import numpy as np


def euclidean_distance(X, Y):
    x = np.linalg.norm(X, axis=1) ** 2
    y = np.linalg.norm(Y, axis=1) ** 2
    x = x[:, np.newaxis]
    y = y[np.newaxis, :]
    s = 2 * (X@Y.T)
    return np.sqrt((abs(x + y - s)))

def cosine_distance(X, Y):
    x = np.linalg.norm(X, axis=1)
    y = np.linalg.norm(Y, axis=1)
    x = x[:, np.newaxis]
    y = y[np.newaxis, :]
    s = (X@Y.T)
    return np.asarray(1 - s/(x*y))

def manhattan_distance(X, Y):
    return np.sum((X - Y), axis=1)