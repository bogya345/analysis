import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def func1():
    return np.arange(10)

def func2(x):
    return x * np.random.randint(0,100)

def main():
    x = func1()
    print(x)
    y = func2(x)
    print(y)
    
    plt.plot(x, y, c='g')
    plt.show()
    pass

if __name__ == '__main__':
    main()
    pass