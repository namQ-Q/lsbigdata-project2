import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1, 21, 10) # 1부터 21까지 랜덤하게 10개의 수를 뽑음음
df.index
model_names = df.index

print(a)
# 두 번째 값 추출
print(a[1])

a[2:5]
a[-2]
a[::2] # 처음부터 끝까지 2칸 건너서 뽑아
a[1:6:2] # 2번쨰 값부터 6번쨰 값까지 2칸 뛰어서 뽑아


# 1에서부터 1000사이 3의 배수의 합은?
x = np.arange(1, 1001, 1)
sum(x[2:1000:3])

# 두 번째 값 제외하고 추출
print(np.delete(a, 1))
np.delete(a, [1, 3]) # a의 2번쨰 4번째 값 제거

a > 3
b = a[a > 3] # 대괄호안에 놀리형을 집어넣으면 true 값만 뽑아온다.
b

np.random.seed(2024)
x = np.random.randint(1, 10000, 300) # 1부터 21까지 랜덤하게 10개의 수를 뽑음음
print(x)
x = x[(x>300) & (x<8000)] # 각 논리형마다 ( )묶어주기
x

!pip install pydataset
import pydataset
df = pydataset.data('mtcars')
mp_df = np.array(df['mpg']) # np array로 변환
# 15 이상 25 이하인 데이터 개수는?
mp_df = mp_df[(mp_df >= 15) & (mp_df <=25)]
len(mp_df)
# 평균 mpg이상인 데이터 갯수는?
sum(mp_df >= mp_df.mean())

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(['A', 'B', 'C', 'F', 'W'])
# a[조건을 만족하는 논리형 벡터]
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)] # a를 만족하는 데이터를 b와 매칭해서 출력

df.index
model_names = df.index
# 연비가 낮은 자동차 모델
model_names[mp_df < mp_df.mean()]

# where 구문 이해하기기
np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a < 50
np.where(a < 50)


np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

#처음으로 5000보다 큰 숫자가 나왔을때, 숫자 위치와 그 숫자는 무엇인가요?
a[np.where(a > 22000)][0]
np.where(a > 22000)[0][0]
x = np.where(a > 22000)
x[0][0] # 첫 [0]에서는 ([array],) 이렇게 튜플로 묶여 있었고, 두 번째 [0]으로 [array]의 첫 원소를 가져옴

# 처음으로 10000보다 큰 숫자가 나왔을때, 50번째로 나온 그 숫자와 위치
x = np.where(a >10000)
x[0][49]
a[81] #21052, 81번째

#500보다 작은 수들 중 가장 마지막에 나온 수
x = np.where(a < 500)
x[0][-1]
a[960]  #391, 961번쨰

# nan
a = np.array([20, np.nan, 13, 24, 309])
a

a + 3 # nan은 계산 안한다.
np.mean(a) # mean으로 못한다.
np.nanmean(a) # nan빼고 평균을 구한다.

# nan과 None 의 차이
False
a = None
b = np.nan
a
b
a+1 # TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
b+1 # nan

# 빈칸을 제거하는 방법
a_filtered = a[~np.isnan(a)]
a_filtered

# 벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec
#리스트는 데이터타입이 달라도 괜찮은데 벡터는 데이터 타입이 같아야함 

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec = np.concatenate([str_vec, mix_vec]) # concatenate는 리스트나 튜플이나 다 가능하다.
combined_vec

col_stacked = np.column_stack((np.arange(1, 5),
                               np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5),
                            np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked # 이렇게 에러가 나는걸 ->
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1
uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked # 세로로 합치기

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
uneven_stacked = np.row_stack((vec1, vec2))
uneven_stacked #가로로 합치기

# 주어진 벡터의 각 요소에 5를 더한 새로운 벡터를 생성하세요.
a = np.array([1, 2, 3, 4, 5])
a + 5

# 주어진 벡터의 홀수 번째 요소만 추출하여 새로운 벡터를 생성하세요.
a = np.array([12, 21, 35, 48, 5])
a[0::2]

# 주어진 벡터에서 최대값을 찾으세요.
a = np.array([1, 22, 93, 64, 54])
a.max()

# 주어진 벡터에서 중복된 값을 제거한 새로운 벡터를 생성하세요.
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

#주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요.
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
x = np.empty(6)
x[1::2] = b
x[0::2] = a
x











