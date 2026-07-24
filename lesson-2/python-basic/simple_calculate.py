import math
import operator
from string import printable
import copy
from sys import prefix
from httpx._transports.default import T


def simple_calculate(a, b, operator):
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/' and b != 0:
        return a / b
    else:
        return None

def shuixianhua_num(num):
    sum=0
    copy_num=num
    while True:
        sum+=(copy_num%10)**3
        copy_num=copy_num//10
        # print(sum, copy_num)
        if copy_num == 0:
            break
    if sum==num:
        return True
    else:
        return False
def main():
    num1=int(input("Enter first number: "))
    num2=int(input("Enter second number: "))
    operator=input("Enter operator: ")
    result=simple_calculate(num1, num2, operator)
    print(result)

def leap_year(year):
    if year % 4 == 0 and year % 100 != 0:
        return True
    elif year % 400 == 0:
        return True
    else:
        return False

def huiwen_str(s):
    for i in range(len(s)//2):
        if s[i] != s[len(s)-1-i]:
            return False
    return True
def math_func():
    a=10
    b=10
    print(id(a), id(b))
    print(a is b)
    c=10000
    d=10000
    print(id(c), id(d))
    print(c is d)

    print(round(3.14159,2))
    print(math.ceil(3.14159))
    print(math.floor(3.14159))
    print(math.sqrt(9))
    print(math.pow(2, 3))

def str_func():
    print(huiwen_str("<[fim-middle]>"))
    print(huiwen_str("abba"))
    print(huiwen_str("abc"))
def list_func():
    #列表的操作
    fruits = ["apple", "banana", "cherry"]

    #添加
    fruits.append("orange")
    fruits.insert(1, "pear")
    fruits.extend(["orange", "pear"])
    #修改
    fruits[0] = "pear"
    print(fruits)

    #查询
    print(fruits.index("banana"))
    print(fruits.count("banana"))
    print("apple" in fruits)
    #删除
    print(fruits)
    fruits.remove("banana")
    poped=fruits.pop()
    print(poped)
    print(fruits)
    fruits.clear()
    print(fruits)
    del fruits
def sorted_list():
    ls1=[2,2,3,1,3]
    ls1.sort()
    print(ls1)

    ls2=[44,55,33,21]
    sort_ls2=sorted(ls2)
    print(ls2,sort_ls2)
def list_copy():
    ls1=[1,2,3,4,5]
    ls3=ls1
    ls2=ls1[0:3]
    ls2[0] = 0
    print(ls1,ls2)
    ls3[0]=10
    print(ls1,ls2)

def list_deep_copy():
    ls1=[1,2,3,[4,5]]
    ls2=copy.deepcopy(ls1)
    ls2[3][0]=10
    print(ls1,ls2)
    ls3=copy.copy(ls1)
    ls3[3][0]=10
    print(ls1,ls3)
def mult_sorted_list():
    menu=[["即可时间",88],["白泽",99]]
    menu.sort(key=operator.itemgetter(1), reverse=True)
    # menu.sort(key=lambda x: x[0])
    print(menu)
def define_tuple():
    tuple1=(10,)
    print(tuple1,type(tuple1))
def use_namedtuple():
    from collections import namedtuple
    Student=namedtuple("Student",["name","age"])
    s = Student("高龄", 31)
    print(s.name,s.age, type(s))
if __name__ == "__main__":
    # main()
    # for i in range(100,100000):
    #     if shuixianhua_num(i):
    #         print(i)
    # print(leap_year(2000))
    # list_func()
    # sorted_list()
    # list_copy()
    # list_deep_copy()
    # mult_sorted_list()
    # define_tuple()
    use_namedtuple()
    odds=[i for i in range(1,11) if i%2==1]
    print(odds)
