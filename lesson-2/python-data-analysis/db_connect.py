import pymysql
import bcrypt
import getpass
class DB:
    def __init__(self, host='localhost', user=None, password=None, database=None, charset='utf8mb4'):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': charset
        }

    def __enter__(self):  
        self.connection = pymysql.connect(**self.config)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.connection.rollback() #有异常回滚
            print(f"An error occurred: {exc_val}")
        else:
            self.connection.commit() #m没有异常提交
        self.cursor.close()
        self.connection.close()
        
    def query(self, sql, params=None):
        self.cursor.execute(sql, params)
        return self.cursor.fetchall()
    
    def execute(self, sql, params=None):
        self.cursor.execute(sql, params)
        return self.cursor.rowcount, self.cursor.lastrowid
    
    def hash_password(self, password):
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def check_password(self, password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
if __name__ == "__main__":
    #从.env文件里读取数据库配置
    import os
    from dotenv import load_dotenv
    load_dotenv()
    db_config = {
        'host': os.getenv('host'),
        'user': os.getenv('user'),
        'password': os.getenv('password'),  
        'database': os.getenv('database'),
        'charset': os.getenv('charset')
    }
    with DB(**db_config) as db:
        data = db.query("SELECT * FROM users")
        print(data)
    #进行注册
    print("开始注册新用户")
    input_username = input("请输入用户名: ")
    #输入秘密这里需要掩盖
    input_password = getpass.getpass("请输入密码: ")
    with DB(**db_config) as db:
        rowcount, last_id = db.execute("INSERT INTO users (username, password, phone) VALUES (%s, %s, %s)", (input_username, db.hash_password(input_password), '1234567890'))
        print(f"Inserted {rowcount} rows, last inserted ID: {last_id}")
    #输入用户名和密码进行验证
    #进行登陆验证这里input可以改成getpass.getpass()来隐藏输入的密码
    username = input("请输入用户名: ")
    password = getpass.getpass("请输入密码: ")
    with DB(**db_config) as db:
        user = db.query("SELECT password FROM users WHERE username=%s", (username,))
        if user:
            hashed_password = user[0][0]
            if db.check_password(password, hashed_password):
                print("登录成功")
            else:
                print("密码错误")
        else:
            print("用户不存在")
    
