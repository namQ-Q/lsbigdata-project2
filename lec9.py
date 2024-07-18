import numpy as np


# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
                          np.arange(12, 16)))
print("행렬:\n", matrix)                   # 세로로 쌓기
matrix = np.vstack((np.arange(1, 5),
                    np.arange(12, 16)))
print("행렬:\n", matrix)                   # 가로로 쌓기
# 행렬의 크기를 재어주는 shape 속성
print("행렬의 크기:", matrix.shape)

# 빈 행렬 만들기 & 채우기기
np.zeros(5)
np.zeros([5,4])                           # 튜플 또는 리스트로 묶어야함

np.arange(1,5).reshape([2, 2])
np.arange(1,7).reshape([2, 3])
np.arange(1,7).reshape([2, -1])           # -1을 통해서 크기를 마음대로 결정할 수 있음

# 0에서부터 99까지의 수 중에서 랜덤하게 50개 숫자를 뽑아서
# 5 by 10 행렬을 만드세요.

np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape(5, -1)
a

np.arange(1, 21).reshape([4,5], order = 'c')          # reshape 기본값은 order = 'c', 가로로 순서대로로
np.arange(1, 21).reshape([4,5], order = 'f')          # order = 'f'는 세로로 순서대로


# 인덱싱
mat_a =np.arange(1, 21).reshape([4,5], order = 'f')
mat_a

mat_a[0, 0]
mat_a[1, 1]
mat_a[2, 3]
mat_a[0:2, 3]                     # 인덱스에는 일반적인 행렬 순서가 아닌 인덱스 번호대로!
mat_a[1:3, 1:4]

mat_a[3,]                         # 행자리 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3, ::2]

# 짝수 행만 선택하려면?
map_b = np.arange(1,101).reshape([20, -1])
map_b[1::2, :]

map_b[[1, 4, 6, 14], :]           # 원하는 행의 인덱스 리스트만 넣어도 가능!

x = np.arange(1, 11).reshape((5, 2)) * 2                         # 5 by 2에서서
filtered_elements = x[[True, True, False, False, True], 0]       # 일차원 벡터으로

map_b[:, 1]            # 벡터
map_b[:, 1:2]          # 행렬



# 필터링
map_b[map_b[:,1] % 7 == 0, :]              # map_b[:, 1]에서 7의 배수의 벡터를 찾아서
                                           # 조건식을 이용해서 true의 행들을 보여준다.



# 사진도 행렬이다.
import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)
# 행렬을 이미지로 표시
plt.show()

plt.imshow(img1, cmap = 'gray', interpolation = 'nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0, 256, 20).reshape(4, -1)
a / 255
plt.imshow(a / 255, cmap = 'gray', interpolation = 'nearest')
plt.colorbar()
plt.show()                                   # 이런식으로 하면 고양이 흑백사진을 만들 수 있다.


# 6의 17 페이지
import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")




# 이미지 읽기
!pip install imageio
import imageio
import numpy as np

jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)          # (88, 50, 4)의 의미는 88 by 50의 행렬이 4장이 있다.
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape            # 첫 번째 장이 행, 열이 변환된다.
                                            # 그렇게 각 장의 R, G, B값을 알 수 있음음

plt.imshow(jelly)
plt.axis('off')                             # 축정보 없애기
plt.show()
jelly

mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)       # 두 개의 2x3 행렬 생성

my_array = np.array([mat1, mat2])           # 3차원 배열로 합치기(곂치기)
my_array.shape                              # (x, y, z) x장 있는데 y by z 행렬이다

my_array2 = np.array([my_array, my_array])
my_array2
my_array2.shape

my_array
filtered_array = my_array[:, :, :-1]
filtered_array                              # 두 번째 차원의 세 번째 요소를 제외한 배열 선택

filtered_array = my_array[:, :, ::2]
filtered_array

filtered_array = my_array[:, [0], :]
filtered_array

filtered_array = my_array[[0], [1], 1:]
filtered_array = my_array[0, 1, 1:]
filtered_array                              # 굳이 괄호를 안해도 되는구나!
                                            # 대신 여러 조건을 넣을때는 해야함 ex. [1, 2]

mat_x = np.arange(1, 101).reshape(5, 5, 4)
mat_x = np.arange(1, 101).reshape(10, 5, 2)
mat_x


# 넘파이 배열 메서드드
a = np.array([[1, 2, 3], [4, 5, 6]])
a
a.sum()
a.sum(axis = 0)
a.sum(axis = 1)

a.mean()
a.mean(axis = 0)
a.mean(axis = 1)

mat_b = np.random.randint(0, 100, 50).reshape(5,-1)
mat_b

# 가장 큰 수
mat_b.max()
mat_b.max(axis = 1)             # 행별로 가장 큰 수
mat_b.max(axis = 0)             # 열별로 가장 큰 수

# 누적해서 더하기
a = np.array([1, 3, 2, 5]).reshape(2,2)
a
a.cumsum()
a.cumsum(axis = 1)              # 행별로 누적합
a.cumsum(axis = 0)              # 열열별로 누적합
                                # 행렬의 차원은 변하지 않는다.
mat_b
mat_b.cumprod()                 # 누적곱

mat_b.reshape(2,5,5).flatten()
mat_b.flatten()                 # 1차원으로 펴준다.

d = np.array([1, 2, 3, 4, 5, 3, 5, 9, 5, 3, 1, 4])
d.clip(2,4)                     # 최대값은 4로 하고 최소값은 2로하는 것으로 바꿔준다.

d.tolist()                      # 배열을 리스트로 바꿔준다.


# 균일확률변수 만들기
np.random.rand(1)

def   X(i):
    return np.random.rand(i) 

X(1)

# 베르누이 확률변수 모수: p 만들어보세요
def Y(p,i):
    x = np.random.rand(i)
    return np.where(x > p, 0, 1)

Y(0.5, 100)
sum(Y(0.5, 100)) / 100
Y(0.5, 1000000000).mean()
    
# 새로운 확률변수  : 가질수 있는 값: 0, 1, 2  /  20% 50% 30%
Y(0.2, 0.5, 3)
np.random.choice([0,1,2], p = [0.2, 0.5, 0.3])
?np.random.choice()                         # 다르게 이해한거
    
p = np.array([0.2, 0.5, 0.3])
def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1,2))
    
Z(p)
    
    











