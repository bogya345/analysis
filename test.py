import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def func1():
    return np.ones(10,10)

def func2(x):
    return np.zeros(10,10)* x

def main():
    x = func1()
    print(x)
    y = func2(x)
    print(y)
    
    plt.plot([1,2,3,4], [1,2,3,4], c='g')
    plt.show()
    pass