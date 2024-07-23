import pandas as pd

fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]


# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()


# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))


# 리스트 변수 변경
range_list[3] = '빅데이터 스쿨'

# 두 번쨰 원소에 ["1st", "2nd", "3rd"] 넣기
range_list[1] = ["1st", "2nd", "3rd"]
range_list
# 두 번쨰 원소의 리스트에서 3rd만 가져오고 싶을 떄
range_list[1][2]              # 앞에서부터 천천히 인덱싱 하면된다.


# 리스트 내포(comprehension)
squares = [x**2 for x in range(10)]    # square 자체는 [ ] 로 묶여있으니 리스트
                                       # 넣고 싶은 수식표현을 x를 사용해서 표현
                                       # for .. in ..구문을 이용하여 원소정보 제공
squares

my_squares = [x**2 for x in [3, 5, 2, 15]]    # in 옆에 내가 원하는 수를 리스트로 넣어도 작동
my_squares

import numpy as np
my_squares2 = [x**2 for x in np.array([3, 5, 2, 15])]
my_squares2

!pip install pandas
import pandas as pd
exam = pd.read_csv("data/exam.csv")


# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]

list1 + list2               # np.array 는 서로 연산되지만 리스트는 합쳐진다.
list1 * 3

numbers = [1, 2, 3]
repeated_list = numbers * 3                                 # 리스트 반복
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)]      # 원소별 반복
repeated_list = [x for x in numbers for _ in [1, 1, 1, 1]]  # 리스트의 원소 수만큼 만복하는구나!
                                                            # 왜냐면 상관없는 문자열은 반복하게
                                                            # 명령내리니까! (라인 120정도에 설명)

# for 루프 문법
# for i in 범위:
#   작동방식
for x in [4, 1, 2, 3]:
  print(x ** 2)

# 리스트를 하나 만들어서
# for 루프를 사용해서 2, 4, 6, 8,..., 20의 수를 채워넣으시오
mylist = []
for x in range(1,11):
  mylist.append(x * 2)               # 방법1

mylist = [0] * 10
for i in range(10):
  mylist[i] = (i + 1) * 2            # 방법2

mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 10
for i range(10):
  mylist[i] = mylist_b[i]


# 퀴즈: mylist_b의 홀수번쨰 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

for x in range(5):
  mylist[x] = mylist_b[x * 2]

mylist



for x in range(1,11):
  for _ in range(2):
    mylist.append(x * 2)



# !참조      _의 의미
# 1. 앞에 나온 값을 가리킴

5 + 4
_ + 6 # _는 앞의 연산인 9를 의미

# 2. 값 생략, 자리 차지
a, _, b = (1, 2, 4)

for x in numbers:
  x
for x in numbers:
  for y in [1, 1, 1, 1]:
    x


# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서.
# for 루프의 : 는 생략한다.
# 실행부분을 먼저 써준다.

mylist = []
for x in range(1,11):
  mylist.append(x * 2)

# ---------------->

mylist = []
mylist.append([x * 2 for x in range(1, 11)])
[x * 2 for x in range(1, 11)]

for i in range(5):
  "hello"                      # i와 상관없는 문자열을 넣으니까 5번 반복한다!

for i in [0, 1, 2]:
  for j in [0, 1]:
    print(i, j)                # 처음에 i가 0으로 고정되면 밑으로 내려가서
                               # j루프를 반복하고 다시 i로 돌아가서서 1로 고정하고 다시 루프

for i in [0, 1]:
  for j in [4, 5, 6]:
    print(i, j)                # 위에꺼 다시 예제

# 리스트 컴프리헨션 변환
numbers = [5, 2, 3]
[i for i in numbers for j in range(4)]


# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

mylist = []
for x in fruits:
  mylist.append(x== "banana")
mylist

# 바나나의 위치를 뱇어내게 하려면?
fruits = ['apple', 'apple', 'banana', 'cherry']
fruits = np.array(fruits)
int(np.where(fruits == 'banana')[0][0])


# 리스트 형식에서 append말고 중간에 원소를 추가하려면
fruits.insert(1, 'pineapple')
fruits
# 리스트에서 원소제거 
fruits.remove('apple')
fruits                       # 모든 apple이 지워지지 않고 첫 apple만 지워진다.
#넘파이로 제거하기
import numpy as np
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])    # 넘파이 배열 생성
items_to_remove = np.array(["banana", "apple"])                           # 제거할 항목 리스트
                                                                          # 넘파이 array가 아니어도 됨(리스트) 
mask = ~np.isin(fruits, items_to_remove)                                  # 불리언 마스크 생성
filtered_fruits = fruits[mask]                                            # 불리언 마스크를 사용하여
                                                                          # 항목 제거
print("remove() 후 배열:", filtered_fruits)













