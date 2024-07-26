# lec14  uniform함수로 복습!

from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
# uniform.rvs(loc=0, scale=1, size=None, random_state=None)    loc = 구간시작점, scale = 구간길이

uniform.rvs(loc = 2, scale = 4, size =1)
k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc = 2, scale = 4)
plt.plot(k, y)
plt.show()

# X~U(2, 6)일 때, p(X < 3.25)
uniform.cdf(3.25, 2, 4)

# X~U(2, 6)일 때, p(5 < X < 8.39)
1 - uniform.cdf(5, 2, 4)

# X~U(2, 6)일 때, 상위 7%의 값은?
uniform.ppf(0.93, 2, 4)

# X~U(2, 6)일 떄, 표본 20개를 뽑고 표본평균 계산
x = uniform.rvs(2, 4, size = 20 * 1000,
                random_state = 42)
x = x.reshape(1000, 20)
blue_x = x.mean(axis = 1)

# 위에꺼 히스토그램 만들기
sns.histplot(blue_x, stat = 'density')
plt.show()
plt.clf()


uniform.var(2, 4)
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(uniform.var(2, 4) / 20))
                                                 # X~N(mu, var)인데     --> var = sigma^2
                                                 # uniform에서 분산을 구하려면
                                                 # Xbar ~ n (mu, sigma^2 / n)
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)
plt.show()
plt.clf()

# uniform.var(a, b-a)     -------> 분산 구하기
# uniform.expect(a, b-a)  -------> 기댓값 구하기











