import numpy as np
import matplotlib.pyplot as plt

import common_dal as dal
import common_funcs as funcs


def h(x, theta):
    # print(theta)
    # print(x[:, 1])
    result = ((x[:, 1] ** 2) - (theta[1] * x[:, 1]) + theta[2])
    n = len(result)
    result = np.array(result).reshape(n, 1)
    # print(result)
    return result


if __name__ == '__main__':
    [x, y], theta = dal.getPreparedData_numpy('storage/simpleTest-1d2_parabola.csv',
                                              #randTheta=True,
                                              theta=np.array([[1], [10], [90]]),
                                              thetaSize=3
                                              )

    # x = funcs.normalize(x)
    x = np.insert(x, [0], np.ones(shape=(len(x), 1)), axis=1)
    x = np.insert(x, [2], np.ones(shape=(len(x), 1)), axis=1)

    firstPred = h(x, theta)
    firstTheta = theta

    # theta, Jhist, thetaHist = funcs.testGradDesc(x, y, theta, h,
    #                                                 alpha=0.00005, iterCount=100000, step=-1)

    theta, Jhist, thetaHist = funcs.steppedGradDesc(x, y, theta, h,
                                                    alpha=0.00005, iterCount=100000, step=-1)


    figure, axes = plt.subplots(nrows=2, ncols=2)
    x1 = x[:, 1]
    axes[0, 0].plot(x1, y, 'gp')
    # axes[0, 0].plot(x1, firstPred, 'rp')
    yP = h(x, theta)
    axes[0, 0].plot(x1, yP, 'b-')

    axes[1, 1].plot(range(0, len(Jhist)), Jhist)

    axes[0, 1].plot(x1, y, 'gp')
    min_index = np.argmin(Jhist)
    bestTheta = thetaHist[min_index]
    axes[0, 1].plot(x1, h(x, bestTheta), 'c-')

    print('first theta\n', firstTheta)
    print('best theta\n', bestTheta)
    print('current theta\n', theta)
    plt.show()
    pass
