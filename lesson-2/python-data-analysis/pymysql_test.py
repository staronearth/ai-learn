import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

def mysql_connect(host='localhost', user=None, password=None, database=None, charset='utf8mb4'):
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset=charset
    )
    return connection

def fetch_data(connection=None):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users")
    data = cursor.fetchall()
    cursor.close()  
    return data

def fetch_data_with_context_manager(connection=None, username=None):
    with connection.cursor() as cursor:
        cursor.execute("SELECT id,username FROM users where username=%s", (username,))
        data = cursor.fetchone()
    return data

def insert_data(connection=None, username=None, password=None, phone=None):
    with connection.cursor() as cursor:
        sql = "INSERT INTO users (username, password, phone) VALUES (%s, %s,%s)"
        cursor.execute(sql, (username, password, phone))
        connection.commit()
        print(f"插入了 {cursor.rowcount} 条数据，最后插入的ID为 {cursor.lastrowid}")

def update_data(connection=None, user_id=None, new_username=None):
    with connection.cursor() as cursor:
        sql = "UPDATE users SET username=%s WHERE id=%s"
        cursor.execute(sql, (new_username, user_id))
        connection.commit()
        print(f"更新了 {cursor.rowcount} 条数据")

def delete_data(connection=None, user_id=None):
    with connection.cursor() as cursor:
        sql = "DELETE FROM users WHERE id=%s"
        cursor.execute(sql, (user_id,))
        connection.commit()
        print(f"删除了 {cursor.rowcount} 条数据")

if __name__ == "__main__":
    #读取.env文件里的数据库配置
    connection = mysql_connect(
        host=os.getenv('host'),
        user=os.getenv('user'),
        password=os.getenv('password'),
        database=os.getenv('database'),
        charset=os.getenv('charset')
    )
    data = fetch_data(connection)
    data2=fetch_data_with_context_manager(connection, username='xiaoming')
    # insert_data(connection, 'xiaoming', 'xiaoming123','1234567890')
    # data = fetch_data(connection)
    # update_data(connection, user_id=1, new_username='new_xiaoming')
    data = fetch_data(connection)
    # delete_data(connection, user_id=1)
    connection.close() 
    for row in data:
        print(row)
    print(data2)
    