import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.arange(33).sum() / 33

# 분산 = var(X)
#      = E[(X-E(X))**2]

np.unique((np.arange(33) - 16) ** 2).sum() * 2 / 33

x = np.arange(33)
(x ** 2).sum() / 33
(x.sum() / 33) ** 2

# var(X) = E[X^2] - E[X] ^2
(x ** 2).sum() / 33 - (x.sum() / 33) ** 2


# X = [0, 1, 2, 3]     P = [1/6, 2/6, 2/6, 1/6] 인 확률변수의 분산구하기
x = np.arange(4)
y = np.array(list)

sum((x**2) * y) - sum(x * y) ** 2
#  E[X^2]         #  E[X] ^2

# X는 0~100 까지의 정수    p는 1-50-1 /2500
# 1-50-1의 의미는 1에서 50에서 다시 1로로
x = np.arange(99)

a = np.arange(1, 51)
b = np.arange(49, 1, -1)

p = np.concatenate((a, b)) / 2500

sum((x**2) * p) - sum(x * y) ** 2


# X = 0, 2, 4, 6    p = 1/6, 2/6, 2/6, 1/6일 때 분산값 구하기기
x = np.arange(0, 7, 2)
p = np.array([1/6, 2/6, 2/6, 1/6])

sum((x**2) * p) - sum(x * y) ** 2


# uniform 한 학률표본을 뽑는데 샘플을 n으로 해서 평균을 구하면(파란벽돌)
# n의 수가 많아질수록 평균분포의 분산이 줄어든다.


9.52**2 / 25
np.sqrt(9.52**2 / 10)
np.sqrt(81/25)

81/25
0.7 * 0.7 + 0.3 * 0.7 * 2 + 0.3 * 0.3
0.3 * 0.7 * 2
0.3 * 0.3

0 + (1/2 * 0.42) + 0.09

from scipy.stats import bernoulli
# 확률질량함수
#bernoulli.pmf(k, p)
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)


bernoulli.cdf(k, p)
bernoulli.ppf(q, p)
bernoulli.rvs(p, size, random_state)

