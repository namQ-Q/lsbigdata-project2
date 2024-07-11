# 데이터 타입
x = 15.34
print(x, "sms ", type(x), "형식입니다.", sep = '')


# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))

b_int = (42)
b_tp = (42,)
type(b_int)
type(b_tp)
b_int = 10
b_int
b_tp = 10
b_tp

# 튜플 생성 예제
a = (10, 20, 30) # a = 10, 20, 30 과 동일
b = (42,)
a[0]

a = [10, 20, 30, 40, 50, 60, 70, 80, 90]
a_tp = (10, 20, 30, 40, 50, 60, 70, 80, 90)

a[1] = 25
a
# a_tp[1] = 25 TypeError: 'tuple' object does not support item assignment

a_tp[3:]
a_tp[:3]
a_tp[1:3]

# 사용자 정의함수
def min_max(numbers):
  return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
result = min_max(a)
result
type(result)


# 딕셔너리 생성 예제
person = {
'name': 'John',
'age': 30,
'city': 'New York'
}
print("Person:", person)

who = {
  'name' : '남규',
  'age' : 28, 
  'home' : ['전주', '수원', '창원']
}

print("who:", who)

who.get('name')
who.get('home')[1]

who_home = who.get('home')
who_home
who_home[0]

# 집합 생성 예제
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨
type(fruits)

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add('hi')
empty_set
empty_set.add('hello')
empty_set
empty_set.add('hi')
empty_set
empty_set.remove('wow')
empty_set
empty_set.remove('bow')
empty_set
empty_set.remove('bye')
empty_set.discard('bye')

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
intersection_fruits = fruits.intersection(other_fruits)
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))
# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

