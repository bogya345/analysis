import pandas as pd
import numpy as np
import random


def getData_numpy(path, usecols=np.arange(0,2)):
    data = np.genfromtxt(fname=path, delimiter=',', skip_header=1, usecols=usecols)
    # data = data[data[:,0].argsort()]
    return data


def getPreparedData_numpy(path, theta=np.array([]), thetaSize=-1, randTheta=False):
    data = getData_numpy(path)
    if thetaSize == -1:
        thetaSize = len(data[0])
    if randTheta:
        theta = np.array([[random.random() for i in range(0, thetaSize)]]).transpose()
    else:
        if not theta.any():
            theta = np.zeros(shape=(thetaSize, 1))
    sortedData = data[data[:, 0].argsort()]
    return getColumns(sortedData), theta


def getColumns(data):
    result = []
    for i in range(0, len(data[0])):
        result.append(np.array([data[:, i]]).transpose())
        continue
    return result


def unitFeatures(*args):
    x = np.array(np.array(args[0]))
    for i in range(1, len(args)):
        x = np.insert(x, [i], args[i], axis=1)
    return x


def getAsFeatures(df):
    result = []
    for i in df:
        item = np.array(df[i])
        item = item.reshape(len(item), 1)
        result.append(item)
    return result


def getDataFrame(path):
    data = pd.read_csv(path)
    res = pd.DataFrame(data)
    print(f'Show head of: {path}')
    print(data.head())
    return res


def saveDataFrame(df, filename='unknown', index=False, header=True):
    dirPath = './storage/saves/'
    path = rf"{dirPath}{filename}.xlsx"
    df.to_excel(path, sheet_name=filename, index=index, header=header)
    print(f'DataFrame saved here: {path}')
    return path
