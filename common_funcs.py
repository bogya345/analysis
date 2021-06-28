
import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    # norm = np.array(x)
    # mu = np.zeros(shape=(len(x), 2))
    # sigma = np.zeros(shape=(len(x), 2))
    #print(len(x))
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    norm = (x - mu) / sigma
    return norm


def costFunc(x, y, theta, hypothesis):
    m = len(y)
    # h_ = h(x, theta)
    # J = (1 / (2 * m)) * ((h_ - y) ** 2).sum()
    # J = (1 / (2 * m)) * ((np.dot(x, theta) - y) ** 2).sum()
    J = (1 / (2 * m)) * ((hypothesis(x, theta) - y) ** 2).sum()
    return J


def difFunc(y, y_pred):
    result = y - y_pred
    return result


def gradDesc(x, y, theta, hypothesis, alpha, iterCount=400):
    x_ = x.transpose()
    m = len(y)
    Jhist = []
    for i in range(0, iterCount):
        error = hypothesis(x, theta) - y
        # error = np.dot(x, theta) - y
        right = np.dot(x_, error)
        theta = theta - ((alpha/m) * right)
        J = costFunc(x, y, theta, hypothesis)
        Jhist.append(J)
    return theta, Jhist


def steppedGradDesc(x, y, theta, hypothesis, alpha, iterCount=400, step=50):
    x_ = x.transpose()
    m = len(y)
    Jhist = []
    thetaHist = []
    for i in range(0, iterCount):
        error = hypothesis(x, theta) - y
        right = np.dot(x_, error)
        theta = theta + ((alpha/m) * right)
        J = costFunc(x, y, theta, hypothesis)
        Jhist.append(J)
        thetaHist.append(theta)
        if i % step == 0 and step != -1:
            figure, axes = plt.subplots(nrows=2, ncols=2)
            x1 = x[:, 1]
            axes[0, 0].plot(x1, y, 'gp')
            axes[0, 0].plot(x1, hypothesis(x, theta), 'b-')
            axes[1, 1].plot(range(0, len(Jhist)), Jhist)
            plt.show()
    return theta, Jhist, thetaHist


def testGradDesc(x, y, theta, hypothesis, alpha, iterCount=400, step=50):
    x_ = x.transpose()
    m = len(y)
    Jhist = []
    thetaHist = []
    for i in range(-100000, 100000):
        error = hypothesis(x, theta) - y
        right = np.dot(x_, error)
        theta = theta - ((alpha/m) * right)
        J = costFunc(x, y, theta, hypothesis)
        Jhist.append(J)
        thetaHist.append(theta)
        if i % step == 0 and step != -1:
            figure, axes = plt.subplots(nrows=2, ncols=2)
            x1 = x[:, 1]
            axes[0, 0].plot(x1, y, 'gp')
            axes[0, 0].plot(x1, hypothesis(x, theta), 'b-')
            axes[1, 1].plot(range(0, len(Jhist)), Jhist)
            plt.show()

    figure, axes = plt.subplots(nrows=2, ncols=2)
    x1 = x[:, 1]
    axes[0, 0].plot(x1, y, 'gp')
    axes[0, 0].plot(x1, hypothesis(x, theta), 'b-')
    axes[1, 1].plot(range(0, len(Jhist)), Jhist)
    plt.show()

    return theta, Jhist