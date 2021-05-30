
import numpy as np
import matplotlib.pyplot as plt
import random

import common_dal as dal
import common_funcs as funcs

def h(x, theta):
    print(x[:,1])
    result = (theta[0] * 1 + theta[1] * x[:,1])
    # result = np.ones(shape=(len(x),))
    result = np.array([random.random()*10 for i in range(0, len(x))])
    result = np.array(result).reshape(len(result), 1)
    return result

if __name__ == '__main__':
    [x1, x2, y], theta = dal.getPreparedData_numpy('./storage/simpleTest-2d.csv')
    x = dal.unitFeatures(x1, x2)

    # x = funcs.normalize(x)
    x = np.insert(x, [0], np.ones(shape=(len(x), 1)), axis=1)

    theta, Jhist = funcs.gradDesc(x, y, theta, h, alpha=0.005, iterCount=400)

    figure, axes = plt.subplots(nrows=2, ncols=2)
    x1 = x[:,1]
    axes[0, 0].plot(x1, y, 'gp')
    axes[0, 0].plot(x1, h(x, theta), 'bp')

    axes[1, 1].plot(range(0, len(Jhist)), Jhist)

    print(theta)
    plt.show()
    pass





