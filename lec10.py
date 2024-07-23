import pandas as pd
import numpy as np

# E[X]
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)

2+ 1 +2 +  1 + 1+ 2


import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)
sum(data < 0.18)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins = 4, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


data_i = np.random.rand(5)

data_mean = data_i.mean()


a = np.random.rand(50000).reshape(-1,5)                     # 랜덤으로 50000개를 reshape한 모양으로 배열
a = np.random.rand(10000, 5).reshape(-1,5)                  # 랜덤으로10000 by 5 행렬로 배열
a
a.mean(axis = 1)

a = np.random.rand(50000).reshape(-1,5) 
aa = a.mean(axis = 1)
aa
plt.clf()
plt.hist(aa, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()














