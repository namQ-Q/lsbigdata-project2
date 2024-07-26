import numpy as np
from scipy.stats import bernoulli

# bernoulli.pmf(k, p)
bernoulli.pmf(1, 0.3)
bernoulli.pmf(0, 0.3)


# 이항분포 p(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
from scipy.stats import binom

binom.pmf(0, n = 2, p = 0.3)
binom.pmf(1, n = 2, p = 0.3)
binom.pmf(2, n = 2, p = 0.3)

# EX. n = 30일 때 각 X의 확률 구하는 리스트
result = [binom.pmf(i, n = 30, p = 0.3) for i in np.arange(31)]
result

import math

math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

# np를 이용해서 구하기기
a = np.arange(1, 55)
a_f = np.cumprod(a)                      # 누적곱을 해서 리스트의 마지막 값을 가져올거임

b = np.arange(1, 27)
b_f = np.cumprod(a)                      # 누적곱을 해서 리스트의 마지막 값을 가져올거임
    
c = np.arange(1, 29)
c_f = np.cumprod(a)                      # 누적곱을 해서 리스트의 마지막 값을 가져올거임
    
a_f[-1] / (b_f[-1] * c_f[-1])            # 값이 너무 커서 결과값 오류가 나니까 로그로 해보자!

#log를 이용한 방법법
a_l = sum(np.log(a))         # log(54!)
b_l = sum(np.log(b))         # log(26!)
c_l = sum(np.log(c))         # log(58!)

a_l -(b_l + c_l)             # log(54! / (26! * 28!))
np.exp(35.168)


math.comb(2,0) * 0.3 ** 0 * (1-0.3) **2
math.comb(2,1) * 0.3 ** 1 * (1-0.3) **1
math.comb(2,2) * 0.3 ** 2 * (1-0.3) **0

# 결국에 위와같은 모든 절차를 binom.pmf(확률질량함수)로 나타낸다.
# binom.pmf(k, n, p)



# X ~ B(n= 10, p = 0.36)일 때, p(X = 4)
binom.pmf(4, 10, 0.36)

# X ~ B(n= 10, p = 0.36)일 때, p(X <=4)
sum(binom.pmf(np.arange(5), 10, 0.36))
binom.cdf(4, 10, 0.36)

# X ~ B(n= 10, p = 0.36)일 때, p(2 < X <=8)
binom.pmf(np.arange(3, 9), 10, 0.36).sum()
binom.cdf(8, 10, 0.36) - binom.cdf(2, 10, 0.36)

# X ~ B(n = 30, p = 0.2)일 떄, p(X < 4 or X >= 25)
1 - binom.pmf(np.arange(4,25), 30, 0.2).sum()
1- binom.cdf(24, 30, 0.2) + binom.cdf(3, 30, 0.2)

a = binom.pmf(np.arange(4), 30, 0.2).sum()
b = binom.pmf(np.arange(25, 31), 30, 0.2).sum()
a + b

# 이걸 어떻게 쓸까?
# 베르누이로 공장의 정상제, 불량제를 구분할 때,
# 불량제를 1이라고 두고 불량률을 p로 둘 때,
# 이항분포(베르놈)를 이용하여 총 제품 10만개를 뽑는다고 할 때
# 불량제가 5000개 이하로 나올 확률을 구할 때?


# rvs함수 (random variates sample) 표본추출확률
# rvs함수는 1이 나올 확률이 0.3인 확률변수에서 하나를 뽑아줘!라는 뜻
bernoulli.rvs(0.3)

# 그러면 X ~ B(n = 2, p = 0.3)을 표현하면
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)      # rvs를 2번 더하면 되지롱!
binom.rvs(n = 2, p = 0.3, size = 10)         # 더하기 치기 귀찮으니까 size를 통해서 원하는 만큼 뽑자!
                                             # 1이 나올 확률이 0.3인데 두 번 뽑아서 1이 나온 수를 10번 반복한다.

# X~B(30, 0.26)일 때, 표본 30개
binom.rvs(n = 30, p = 0.26, size = 30)

# X~B(30, 0.26)일 때, X의 기댓값은?      ------------>  E[X] = n*p
(np.arange(31) * binom.pmf(np.arange(31), 30, 0.26)).sum() / 31        # 오답!
# 따라서 이항분포의 기댓값은 np이다.

# X~B(30, 0.26)을 시각화 해보세요!
X = np.arange(0, 31)
P = binom.pmf(np.arange(31), 30, 0.26)

import matplotlib.pyplot as plt

plt.bar(X, P)
plt.show()
plt.clf()

# 책처럼 하기
import pandas as pd
df = pd.DataFrame({'X label' : X,
                   'Y label' : P})
import seaborn as sns
sns.barplot(data = df, x = 'X label', y = 'Y label')
plt.show()


# cdf: culmulative dist function
# (누적확률분포 함수)
# F_X(x)


# X~B(30, 0.26)일 때, p(x <= 4) 일 경우
binom.cdf(4, 30, 0.26)                           # binom.pmf(k, n, p).sum()을 알아서 계산해준다

# X~B(30, 0.26)일 때, p(4 < x <= 18) 일 경우
binom.cdf(18, 30, 0.26) - binom.cdf(4, 30, 0.26)

# X~B(30, 0.26)일 때, p(13< x < 20)
binom.cdf(19, 30, 0.26) - binom.cdf(13, 30, 0.26)


# 
x_1 = binom.rvs(30, 0.26, size = 1)
x = np.arange(31)
prob_x = binom.pmf(x, 30, 0.26)
sns.barplot(prob_x, color = 'blue')
plt.scatter(x_1, 0.002, color = 'red', zorder = 10, s =5)
plt.axvline(7.8, color = 'red')
plt.show()
plt.clf()                                # 확률질량함수 그래프로 시각화하고
                                         # 랜덤으로 하나 뽑아서 점 찍는 방법

x_1 = binom.rvs(30, 0.26, size = 3)
x = np.arange(31)
prob_x = binom.pmf(x, 30, 0.26)
sns.barplot(prob_x, color = 'blue')
plt.scatter(x_1, np.repeat(0.002, 3), color = 'red', zorder = 10, s =5)
plt.axvline(7.8, color = 'red')
plt.show()
plt.clf()                                # 확률질량함수 그래프로 시각화하고
                                         # 랜덤으로 여러개 뽑아서 점 찍는 방법



# 정리
# pmf = p(X = k)
# cdf = p(X <= k)
# rvs = 랜덤샘플사이즈

# X~B(n, p) - 앞면(1)이 나올 확률이 p인 동전을 n번 던져서 나온 앞면의 수


# 마지막 ppf
# binom.cdf(q, n, p)
# cdf가 0에서 k까지 나오는 확률을 구하고 싶은거라면
# ppf는 확률 p가 나오는 k값을 알고싶을 때

# p(x < 0.5)
binom.ppf(0.5, 30, 0.26)
binom.cdf(8, 30, 0.26)
binom.cdf(7, 30, 0.26)

# p(x < 0.7)
binom.ppf(0.7, 30, 0.26)


# mu가 1이고 sigma가 0인 정규분포 식
def a(x):
    return (1 / np.sqrt(2 * math.pi)) * math.exp(x ** 2 / -2)
    
a(0)

from scipy.stats import norm
# norm.pdf(x, loc = mu, scale = sigma)             # loc = mu, scale = sigma
norm.pdf(0, loc = 0, scale = 1)

# mu = 3, sigma = 4, x =5인 정규분포
norm.pdf(5, 3, 4)


# 정규분포 pdf 그리기
k = np.linspace(-3, 3, 5)
y = norm.pdf(np.linspace(-3, 3, 5), loc = 0, scale = 1)
plt.scatter(k, y)
plt.show()
plt.clf()                                          # 기본원리


k = np.linspace(-3, 3, 100)
y = norm.pdf(k, loc = 0, scale = 1)
# plt.scatter(k, y, s=1)
plt.plot(k, y)
plt.show()
plt.clf()                                          # 기본원리 2


k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = -3, scale = 1)
plt.plot(k, y)
plt.show()
plt.clf()                                          # mu는 분포의 중심을 정하는 변수구나!


k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc = 0, scale = 1)
y2 = norm.pdf(k, loc = 0, scale = 2)
y3 = norm.pdf(k, loc = 0, scale = 0.5)
plt.plot(k, y)
plt.plot(k, y2)
plt.plot(k, y3)
plt.show()
plt.clf()                                          # sigma는 분포의 퍼짐을 정하는 변수구나!
                                                   # sigma가 작아지면 평균근처에 분포한다.


norm.cdf(0, 0, 1)                                  # k값을 0으로 하면 0.5구나
norm.cdf(100, 0, 1)                                # k값을 100으로 하면 대략적으로나마
                                                   # 정규분포 면적이 1이라는걸 알 수 있다.

# p(-2< x < 0.54)의 확률은?
norm.cdf(0.54, 0, 1) - norm.cdf(-2, 0, 1)

# p(x < 1 or x > 3)
1 - norm.cdf(3, 0, 1) + norm.cdf(1, 0, 1)

# X~N(3, 5^2)일 때,  p(3 < x <5)
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)

# 위에가 진짜 맞는지 확인
# X~N(3, 5^2)일 때, p(3 < x < 5) =? 15.54%
# 위 확률변수에서 표본 1000개 뽑아보자!
x = norm.rvs(3, 5, 1000)
sum((x > 3) & (x < 5)) / 1000

# X~N(0, 1)일 때 1000개 뽑아서 진짜 0보다 작은수가 50%인지 확인
x = norm.rvs(0, 1, 1000)
np.mean(x < 0)


x = norm.rvs(3, 2, 1000)
x
sns.histplot(x)
plt.show()
plt.clf()

x = norm.rvs(3, 2, 1000)
x
sns.histplot(x, stat = 'density')                         # density를 안쓰면 히스토그램은 빈도수를 보여주고
                                                          # density를 쓰면 밀도를 보여준다.
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 3, scale = 2)
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)
plt.show()
plt.clf()





    
