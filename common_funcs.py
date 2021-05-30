
import numpy as np


def h(x, theta):
    # x, theta = common.transformToNpArrays(x, theta)
    # result = (x[:,0] * theta[0] + x[:,1] * theta[1])
    print(x[:,1])
    result = (theta[0] * 1 + theta[1] * x[:,1])
    result = np.array(result).reshape(len(result), 1)
    return result


def normalize(x):
    # norm = np.array(x)
    # mu = np.zeros(shape=(len(x), 2))
    # sigma = np.zeros(shape=(len(x), 2))
    print(len(x))
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    norm = (x - mu) / sigma
    return norm


def costFunc(x, y, theta):
    m = len(y)
    # h_ = h(x, theta)
    # J = (1 / (2 * m)) * ((h_ - y) ** 2).sum()
    J = (1 / (2 * m)) * ((np.dot(x, theta) - y) ** 2).sum()
    return J


def difFunc(y, y_pred):
    result = y - y_pred
    return result


def gradDesc(x, y, theta, alpha, iterCount=400):
    x_ = x.transpose()
    m = len(y)
    Jhist = []
    for i in range(0, iterCount):
        error = h(x, theta) - y
        # error = np.dot(x, theta) - y
        right = np.dot(x_, error)
        theta = theta - ((alpha/m) * right)
        J = costFunc(x, y, theta)
        Jhist.append(J)
    return theta, Jhist