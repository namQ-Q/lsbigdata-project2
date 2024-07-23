import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 산점도
mpg = pd.read_csv('data/mpg.csv')
mpg

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy')
plt.show()
plt.clf()

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy') \
   .set(xlim = [3, 6], ylim = [10,30])
plt.show()
plt.clf()

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy', hue = 'drv')
plt.show()
plt.clf()

# 막대그래프
mpg['drv'].unique()

df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(mean_hwy = ('hwy', 'mean'))
df_mpg
sns.barplot(data = df_mpg, x = 'drv', y = 'mean_hwy', hue = 'drv')
plt.show()
plt.clf()

sns.barplot(data = df_mpg.sort_values('mean_hwy'),
                   x = 'drv', y = 'mean_hwy', hue = 'drv')
plt.show()
plt.clf()

df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))
df_mpg
sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()
plt.clf()
sns.countplot(data = mpg, x = 'drv', hue = 'drv')
plt.show()
plt.clf()                              # countplot과 barplot의 차이점
                                       # countplot읜 빈도표 만들 작업을 생략하고
                                       # 원자료를 곧바로 사용한다.
                                       # 자동적으로 sorting하지는 않는다.
mpg['drv'].unique()
df_mpg['drv'].unique()



