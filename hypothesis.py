
import numpy as np

def h(x, theta):
    # x, theta = common.transformToNpArrays(x, theta)
    # result = (x[:,0] * theta[0] + x[:,1] * theta[1])
    print(x[:,1])
    result = (theta[0] * 1 + theta[1] * x[:,1])
    result = np.array(result).reshape(len(result), 1)
    return result