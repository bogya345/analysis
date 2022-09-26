import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from test_imp import func1

def func2(x):
    return x * np.random.randint(0,100)

def main():

    x = func1()
    print(x)

    y = func2(x)
    print(y)
    
    plt.plot(x, y, c='g')

    plt.savefig('./figures/foo.png')
    plt.savefig('./figures/foo.pdf')

    plt.show()
    pass

if __name__ == '__main__':
    main()
    pass