a = (1, 2, 3)
a

a = [1, 2, 3]
a

# soft copy
b = a
b

a[1] = 4
a

b
id(a)
id(b) # 서로 같아서 a를 바꾸면 b도 같이 바낀다.

# deep copy
a = [1, 2, 3]
a

id(a)

b = a[:]
b = a.copy()

id(b) # id가 서로 다르기 떄문에 a를 바꿔도 b가 바뀌지 않는다.

a[1] = 4
a
b


# 수학함수
import math
x = 4
math.sqrt(x)

exp_val = math.exp(5)
exp_val

log_val = math.log(10, 10)
log_val

# 복잡한 변수 만들기 예시(확률밀도함수)

def normal_pdf(x, mu, sigma):
sqrt_two_pi = math.sqrt(2 * math.pi)
factor = 1 / (sigma * sqrt_two_pi)
return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mycode(x, y, z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

mycode(2, 9, math.pi / 2)


def my_c(x):
  return math.cos(x) + math.sin(x) * math.exp(x)

my_c(math.pi)

# snippet 만들기(일종의 단축키!)

def fname(`indent('.') ? 'self' : ''`):
  """docstring for fname"""
  # TODO: write code...

def   (input):
    contents
    return 

# 넘파이를 이용한 벡터

!pip install numpy
import numpy as np

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)    

type(a)
a[3]    
a[2:5]    
a[1:4]    

# 빈 배열 생성
x = np.empty(3)
print("빈 벡터 생성하기:", x)
# 배열 채우기
x[0] = 3
x[1] = 5
x[2] = 3
print("채워진 벡터:", x)
    
b = np.empty(3)
b    
b[0] = 1    
b[1] = 4    
b[2] = 10    
b    

vec1 = np.array([1, 2, 3, 4, 5])    
vec1 = np.arange(100)    
vec1 = np.arange(1, 101, 0.5)    
vec1    
# np.arange(시작값, 끝값보다 높은 수, 간격)    

l_space1 = np.linspace(1, 100, 100)
l_space1
# np.linspace(시작값, 끝값, 데이터 수)

vec1 = np.arange(5)
np.repeat(vec1, 5)

# repeat vs tile
vec1= np.arange(5)
np.repeat(vec1, 3) # array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
np.tile(vec1, 3) # array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

vec1 + vec1

max(vec1)
sum(vec1)

# 문제: 35672 이하 홀수들의 합?

vec_5 = np.arange(1, 35673, 2)
vec_5
sum(vec_5)
vec_5.sum()

vec_5.shape
len(vec_5)
vec_5.len()                         #AttributeError: 'numpy.ndarray' object has no attribute 'len'

b = np.array([[1, 2, 3], [4, 5, 6]])
len(b) # 첫 번쨰 차원의 길이
b.shape # 각 차원의 크기
b.size # 전체 요소의 갯수


a = np.array([1, 2])
b = np.array([1, 2, 3, 4])
a + b                               # ValueError: operands could not be broadcast together with shapes (2,) (5,)

np.tile(a,2) + b
np.repeat(a,2) + b

b == 3                              # array([False, False,  True, False])


# 문제! 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?

x = np.arange(1, 35672)
sum(x % 7 == 3)

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape
b.shape

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
[10.0, 10.0, 10.0],
[20.0, 20.0, 20.0],
[30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
# 두 shape의 숫자가 같이 공유를 한다면 벡터를 세로벡터로 바꿔준 후, shape을 맞춰 더해주면 브로드캐스트가 작동하게 된다.
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector
vector.shape
result = matrix + vector
result







