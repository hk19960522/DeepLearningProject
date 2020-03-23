import numpy as np
import matplotlib.pyplot as plt
import os
import random


def draw_graph(filenames):
    x_data, y_data = [], []
    for filename in filenames:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cnt = 1
                x, y = [], []
                line = f.readline()
                while line:
                    if line[0:6] == 'tensor':
                        real = float(line[7: -2])
                        for _ in range(0, 1):
                            x.append(cnt)
                            cnt += 1
                            delta = real * 0
                            y.append(real + random.uniform(-delta, delta) + 1.0)
                    else:
                        x.append(cnt)
                        y.append(float(line))
                        cnt += 1
                    line = f.readline()
                x_data = x
                if len(y_data) > 0:
                    for _y_data, _y in zip(y_data, y):
                        _y_data.append(_y)
                else:
                    for _y in y:
                        y_data.append([_y])


    plots = plt.plot(x_data, y_data)
    plt.legend(plots, ('loss'))
    plt.show()




# filenames = ['../result/Err.txt', '../result/F_Err.txt']
filenames = ['../result/Loss.txt']
draw_graph(filenames)

'''
x = [i for i in range(0, 10)]
y = np.random.randn(10, 1).cumprod(0)

plt.plot(x, y)
plt.show()
'''


