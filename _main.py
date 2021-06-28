import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import common_dal as dal
import common_funcs as funcs

def h(x, theta):
    # x, theta = common.transformToNpArrays(x, theta)
    # result = (x[:,0] * theta[0] + x[:,1] * theta[1])
    print(x[:,1])
    result = (theta[0] * 1 + theta[1] * x[:,1])
    result = np.array(result).reshape(len(result), 1)
    return result


tmp = dal.getDataFrame('./storage/test.csv')
x1, x2, y = dal.getAsFeatures(tmp)
x = np.array(x1)
x = np.insert(x, [1], x2, axis=1)
theta = np.array([[0], [0], [0]])

print(x)
print(x[0][1])
x = funcs.normalize(x)
x = np.insert(x, [0], np.ones(shape=(len(x1),1)), axis=1)

figure, axes = plt.subplots(nrows=2, ncols=2)

theta, Jhist = funcs.gradDesc(x, y, theta, h, alpha=0.005)

axes[0, 0].plot(x[:,1], y, 'gp')
axes[0, 0].plot(x[:,1], h(x, theta), 'b-')
axes[1, 1].plot(range(0, len(Jhist)), Jhist)

plt.show()

# tracer = pd.DataFrame()
# colsCount = 3
# for i in range(0, 10):
#     y_pred = h(x, theta)
#     c = costFunc(x, y, theta)
#     d = difFunc(y, y_pred)
#
#     tracer.insert(i*colsCount, f'y{i}', y[:,0])
#     tracer.insert(i*colsCount+1, f'yP{i}', y_pred[:,0])
#     tracer.insert(i*colsCount+2, f'd{i}', d[:,0])
#     print('costFunc = ', costFunc(x, y, theta))
#     print(tracer)
#
#
#     continue
# axes[0, 0].plot(x, y, 'gp')
# axes[0, 0].plot(x, tracer['yP0'], 'rp')
#
# axes[0, 1].plot(x, y, 'gp')
# axes[0, 1].plot(x, tracer['yP9'], 'rp')
#
# plt.show()
