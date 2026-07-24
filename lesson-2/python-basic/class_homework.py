from random import seed


class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.__is_borrowed = False
        self.__borrowed_by = None

    @property
    def is_borrowed(self):
        return self.__is_borrowed

    @property
    def borrowed_by(self):
        return self.__borrowed_by

    @borrowed_by.setter
    def borrowed_by(self, value):
        self.__is_borrowed = True
        self.__borrowed_by = value

    def __str__(self) -> str:
        if self.__is_borrowed:
            return f"{self.title} is borrowed by {self.__borrowed_by}"
        return f"{self.title} is not borrowed"

class Person:
    def __init__(self,name,age) -> None:
        self.name = name
        self.age = age

class Student(Person):
    def __init__(self, name, age, student_id) -> None:
        super().__init__(name, age)
        self.student_id = student_id
        self.books=[]

    def borrow_book(self, book):
        if not book.is_borrowed:
            book.borrowed_by = self
            self.books.append(book)
        else:
            print(f"{book.title} is already borrowed")
    def my_books(self):
        print(f"{self.name} 的书籍:")
        for book in self.books:
            if book.borrowed_by == self:
                print(book)

    def __str__(self) -> str:
        return f"Student {self.name}, ID: {self.student_id}"

class Admin(Person):
    def __init__(self, name, age, admin_id) -> None:
        super().__init__(name, age)
        self.admin_id = admin_id

    def check_all_books(self, books):
        print("所有书籍的借阅状态:")
        for book in books:
            print(book)

if __name__ == "__main__":
    book1 = Book("The Great Gatsby", "F. Scott Fitzgerald")
    book2 = Book("1984", "George Orwell")
    book3 = Book("钢铁是怎样炼成的", "保尔")
    student = Student("John", 20, "S001")
    student2 = Student("Alex", 22, "S002")
    admin = Admin("Admin", 30, "A001")
    student.borrow_book(book1)
    student2.borrow_book(book2)
    student2.borrow_book(book1)
    admin.check_all_books([book1, book2, book3])
    student.my_books()
    student2.my_books()
