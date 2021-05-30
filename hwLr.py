
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression

import common_dal as cmn

def linearRegresion(x, y):
    nx = len(x)
    ny = len(y)
    if nx != ny:
        return -1, -1
    n = nx

    x = np.array(x)
    x.reshape(1, n)
    y = np.array(y)
    y.reshape(n, 1)

    sum_y = y.sum()
    sum_x2 = (x**2).sum()
    sum2_x = x.sum()**2
    sum_x = x.sum()
    sum_xy = (x*y).sum()

    a_up = (sum_y * sum_x2) - (sum_x * sum_xy)
    a_down = (n * sum_x2) - sum2_x
    a = a_up / a_down

    b_up = (n * sum_xy) - (sum_x * sum_y)
    b_down = (n * sum_x2) - sum2_x
    b = b_up / b_down

    return a, b

if __name__ == '__main__':

    figure, axes = plt.subplots(nrows=1, ncols=2)
    # figure(figsize=(8, 6), dpi=80)

    df = cmn.getDataFrame('./storage/data.csv')

    X = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression

    Y_pred = linear_regressor.predict(X)  # make predictions

    axes[0].scatter(X, Y)
    axes[0].plot(X, Y_pred, color='red')

    x = df['Height']
    y = df['Weight']
    a, b = linearRegresion(x, y)
    y_suggest = []
    for i in x:
        el = a + (b * i)
        y_suggest.append(el)
        continue
    axes[1].plot(x, y, 'gp')
    axes[1].plot(x, y_suggest, 'b-')

    plt.show()

    pass