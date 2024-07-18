import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 합치기
test1 = pd.DataFrame({'id'      : [1, 2, 3, 4, 5],
                      'midterm' : [60, 80, 70, 90, 85]})
test2 = pd.DataFrame({'id'      : [1, 2, 3, 4, 5],
                      'final'   : [70, 83, 65, 95, 80]})
test1
test2

total = pd.merge(test1, test2, how = 'left', on = 'id')
total                                           # left join


test1 = pd.DataFrame({'id'      : [1, 2, 3, 4, 5],
                      'midterm' : [60, 80, 70, 90, 85]})
test2 = pd.DataFrame({'id'      : [1, 2, 3, 40, 5],
                      'final'   : [70, 83, 65, 95, 80]})

total = pd.merge(test1, test2, how = 'left', on = 'id')
total                                    # 기준이 test1이니까 4는 nan 40은 none

total = pd.merge(test1, test2, how = 'right', on = 'id')
total                                    # 기준이 test2라서 40은 nan 4는 none

total = pd.merge(test1, test2, how = 'inner', on = 'id')
total                                    # 두 기준이 공통적으로 가지고 있는 데이터로 join(교집합)
                                         # 필터링의 기준을 같이 가지고 있음

total = pd.merge(test1, test2, how = 'outer', on = 'id')
total                                    # 두 기준이 가지고 있는 모든 데이터를 join(합집합)


name = pd.DataFrame({'nclass'  : [1, 2, 3, 4, 5],
                     'teacher' : ['kim', 'lee', 'park', 'choi', 'jung']})
name
exam = pd.read_csv('data/exam.csv')
exam
exam_new = pd.merge(exam, name, how = 'left', on = 'nclass')
exam_new




# 세로로 합치기
score1 = pd.DataFrame({'id'      : [1, 2, 3, 4, 5],
                      'score' : [60, 80, 70, 90, 85]})
score2 = pd.DataFrame({'id'      : [6, 7, 8, 9, 10],
                      'score'   : [70, 83, 65, 95, 80]})
score1
score2
score_all = pd.concat([score1, score2])
score_all




# 결측치찾기
df = pd.DataFrame({'sex'   : ['M', 'F', np.nan, 'M', 'F'],
                   'score' : [5, 4, 3, 4, np.nan]})
df
pd.isna(df)





# 결측치 제거하기
df.dropna(subset = 'score')               # socre를 기준으로 nan를 제거한다.
df.dropna(subset = ['score', 'sex'])      # 여러 기준으로 nan을 제거하려면 시리즈로 구성한다.
df.dropna()                               # 아묻따 nan제거, 변수 중 하나라도 있으면 모든 행 제거

exam.loc[[2, 7, 14], ['math']] = np.nan
exam.iloc[[2, 7, 14], [2]] = 3
exam

df
df.loc[df['score'] == 3.0,['score']] = 4  # 원하는 값을 필터링해서 값을 변환하는 법법
df[df['score'] == 3.0]
df['score'] = 4
df





# 수학점수 50점 이하인 학생들 점수 50점으로 상향 조정!
exam
exam.loc[exam['math'] <= 50 , 'math'] = 50
exam

# 영어점수 90점 이상 90으로 하향 조정 iloc사용
exam.iloc[exam['english'] >= 90, 3] = 90
exam

exam.iloc[exam[exam['english'] >= 90].index, 3]

# iloc는 숫자벡터가 들어가야함...
# np.where로 넣으려고 해도 튜플이라서 실행 안됨
# 실행 시키려면 [0]를 넣어서 튜플 안에 있는 np.array를 꺼내온다.
# index 벡터는 잘 작동함함

# math 점수 50점 이하 -로 변경
exam.loc[exam['math'] <= 50, 'math'] = '-'
exam

# '-' 결측치를 수학점수 평균 바꾸고 싶은 경우
exam.loc[exam['math'] == '-', 'math'] = np.nan
exam.loc[pd.isna(exam['math']), 'math'] = exam['math'].mean()   # 한 번 더 확인인
exam
exam.astype('math', float)
# 다른 사람들이 mean만든 방법법
exam.loc[exam['math'] != '-', 'math'].mean()     # 1
exam.query('math not in ["-"]')['math'].mean()   # 2

vector = np.array([np.nan if x == '-' else float(x) for x in exam["math"]])
vector = np.array([float(x) if x != '-' else np.nan for x in exam["math"]])

np.nanmean(vector)












