class Person:
    def __init__(self, name,age) -> None:
        self.name = name
        self.age = age


class Student(Person):
    def __init__(self,name,age) -> None:
        super().__init__(name, age)
        self.__score=0
    def __str__(self) -> str:
        return f"Student(name={self.name}, age={self.age}, score={self.__score})"
    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        if not 0 <= value <= 100:
            raise ValueError("score must be between 0 and 100")
        self.__score = value

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        # 使用 !r 转换，确保 x,y 的 repr() 被正确显示（比如字符串会带引号）
        return f"{self.__class__.__name__}({self.x!r}, {self.y!r})"

    def __str__(self):
        return f"坐标: ({self.x}, {self.y})"
if __name__ == "__main__":
    s = Student("Alice", 20)
    s.score = 90
    print(s.score)
    print(s)
    # s._Student__score = 120 这个是隐藏是约定不是绝对的
    # s.score=120
    print(s.score)
    print(repr(s))

    p = Point(10, 20)
    print(p)
    print(repr(p))
