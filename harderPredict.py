
import common_dal

import numpy as np


def h(x, thetas):
    result = thetas[0] + thetas[1] * x
    return result


def costFunc(x, y, theta):
    x, y, theta = common_dal.transformToNpArrays(x, y, theta)
    m = len(y)
    result = (1/(2*m)) * (((x*theta)-y)**2).sum()
    return result


def gradientDesc(x, y, theta, alpha, iterations):
    x, y, theta = common_dal.transformToNpArrays(x, y, theta)
    y = y.transpose()
    theta = theta.transpose()
    m = len(y)
    Jhist = np.zeros(iterations)

    for i in range(0, iterations):
        error = (x * theta) - y
        theta = theta - ((alpha/m) * x.transpose() * error)

        Jhist[i] = costFunc(x, y, theta)
        continue

    return theta, Jhist


if __name__ == '__main__':
    df = common_dal.getDataFrame('./storage/data.csv');
    x = df['Height']
    y = df['Weight']

    theta = np.array([1, 1]).transpose()
    m = len(y)  # number of training examples
    alpha = 0.4
    iterations = 10

    theta, Jhist = gradientDesc(x, y, theta, alpha, iterations)

    y_hyp = np.zeros()

    for i in range(0, len(x)):
        hyp = theta[0] * 1 + theta[1] * x[i]
        y_hyp.append()

    y_hyp = np.array(y_hyp).transpose()

    print(y_hyp)

    pass