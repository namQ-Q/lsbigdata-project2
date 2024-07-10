a=10
a
a='안녕하세요!'
a
a="hi!"
a=[1,2,3]
a
b=[4,5,6]
b
a+b

str1='LS빅데이터스쿨'
str2='화이팅!'
str1+' '+str2


print(a)

a=10
b=3.3
prinb('a + b = ',a+b)
print('a - b = ',a-b)
print('a * b = ',a*b)
print('a / b = ',a/b)
print('a % b = ',a%b)
print('a // b = ',a//b)
print('a ** b = ',a**b)

(a**b) // 7
(a**b) // 7

a == b
a != b
a < b
a > b
a <= b
a >= b

ex1 = (2 ** 4 + (12453 // 7)) % 8
ex2 = (9 ** 7 / 12) * (36452 % 253)


user_age = 25
is_adult = user_age >= 18
print("성인입니까?", is_adult)

TRUE = '정답'
a = 'True'
b = TRUE
#TRUE 변수로 지정되어있지 않다.그래서 error.
c = true
d = True

TRUE

a = True
b = False

a and b
a or b
not b

# True: 1
# False:0
True + True
True + False
False + True
False + False

# and 연산자          # True와 False의 사칙연산으로 따지면 곱과 같음
True and False
True and True
False and True
False and False
True * True
True * False
False * True
False * False

# or 연산자         # True와 False의 사칙연산으로 따지면 합과 같음(얼추)
True or False
True or True
False or True
False or False
a or b
min(a + b, 1)

# and는 포함하는 데이터 or은 이거나 하는 데이터

a = 3
a = a + 10
a += 10
a

a -= 4
a

a %= 3
a

a += 12
a

a **= 2
a

str1 = 'hello'
str1 + str1
str1 * 3
  
# 문자열 변수 할당
str1 = "Hello! "
# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

# binary
bin(5)
bin(-5)
# '0b'는 이진수를 나타내는 기호

bin(6)
bin(-6)


x = 13
~x
bin(13)
bin(-14)

x = 3
bin(x)
bin(~x)

x = 16
~x

x = 129
~x

x = 128
~x

x = -5
~x

x=3
~x

x = -4
~x

pip install pydataset
# 오류가 뜨는 이유는 파이썬과 pip는 별개의 프로그램이기 때문이다.
# 그러니 밖에서 돌려야 하므로 터미널에서 쓰던가 앞에 !를 붙이던가

import pydataset
pydataset.data()
# 함수임을 알 수 있는 방법은 ()
pydataset.data().head(10)
df = pydataset.data('AirPassengers')
df
df.head()
df.head(13)

