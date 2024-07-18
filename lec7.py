import pandas as pd
import numpy as np
exam = pd.read_csv('data/exam.csv')

# 데이터 탐색
exam.head()
exam.tail()
exam.shape
exam.shape()  # 메서드 vs 속성(어트리뷰트)
              # 메서드는  괄호가 있고 속성은 없다.
              # 속성은 값이 튜플로 값이 나온다.
exam.info()
exam.describe()

# 내장함수는 아무런 도움 없이 작동되는 함수
# 패키지함수는 패키지의 도움이 필요함(패키지 로드 같은거)
# 객체(ex. df)를 만들어야 함수를 작동

type(exam)       # <class 'pandas.core.frame.DataFrame'
var = [1, 2, 3]
type(var)        # <class 'list'>


# 변수명 바꾸기
exam2 = exam.copy()
exam2 = exam2.rename(columns = {'nclass' : 'class'})


# 파생변수 만들기
exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']
exam2.head()
exam2['test'] = np.where(exam2['total'] >= 200, 'pass', 'fail')
exam2.head()
count_test = exam2['test'].value_counts()
count_test.plot.bar(rot = 0)
?.plot.bar()

import matplotlib.pyplot as plt
plt.show()
plt.clf()

exam2['test_2'] = np.where(exam2['total'] >= 200, 'A',
                  np.where(exam2['total'] >= 100, 'B', 'C'))
exam2.head()

exam2['test_2'].isin(['A', 'C'])


# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam.query('nclass == 1')
exam[exam['nclass'] == 1]

exam.query('math > 50')
exam.query('math < 50')
exam.query('english >= 50')
exam.query('english <= 80')
exam.query('nclass == 1 & math >= 50')
exam.query('nclass == 1 and math >= 50')
exam.query('nclass == 1 | nclass == 3')
exam.query('nclass == 1 or nclass == 3')
exam.query('nclass in [1, 3, 5]')
exam.query('nclass not in [1, 3, 5]')

# exam['nclass'].isin([1,2])
# ~exam['nclass'].isin([1,2])


# 필요한 변수 추출하기
exam['nclass']
exam[['nclass']]                                # 데이터 프레임 형식으로 가져온다.
exam[['id', 'nclass']]

exam.query('nclass == 1')[['math', 'english']]
exam.query('nclass == 1') \
     [['math', 'english']] \
     .head()


# 정렬하기
exam.sort_values('math')
exam.sort_values('math', ascending = False)
exam.sort_values(['nclass', 'english'], ascending = [True, False])


# 변수추가
exam.assign(total = exam['math'] + exam['english'] + exam['science'],
            mean = (exam['math'] + exam['english'] + exam['science']) /3) \
    .sort_values('total', ascending = False) \
    .head()


# lambda 함수 사용하기
exam.assign(total = lambda x : x['math'] + x['english'] + x['science'],
            mean = lambda x : (x['math'] + x['english'] + x['science']) /3) \
    .sort_values('total', ascending = False) \
    .head()

exam.assign(total = lambda x : x['math'] + x['english'] + x['science'],
            mean = lambda x : x['total'] /3) \
    .sort_values('total', ascending = False) \
    .head()                                            # lambda를 이용하면 assign내에서 만든 함수를 곧바로 쓸 수 있다.



# 요약하는 .agg()
exam2 = pd.read_csv('data/exam.csv')

exam2.agg(mean_math = ('math', 'mean'))
exam2.groupby('nclass') \
     .agg(mean_math = ('math', 'mean'))
exam2.groupby('nclass', as_index = False) \
     .agg(mean_math = ('math', 'mean'))                # 데이터프레임 형식으로 나온다.
     
exam.groupby('nclass', as_index = False) \
    .agg(mean_math = ('math', 'mean'),
         sum_math = ('math', 'sum'))                   # 여러 통계량을 한 번에 구할 수 있다.


# 166p 문제
mpg = pd.read_csv('data/mpg.csv')

mpg.query('category == "suv"') \
   .assign(total = (mpg['cty'] + mpg['hwy']) / 2) \
   .groupby('manufacturer') \
   .agg(mean_tot = ('total', 'mean')) \
   .sort_values('mean_tot', ascending = False) \
   .head()










