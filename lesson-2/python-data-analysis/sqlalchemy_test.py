from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, ForeignKey, create_engine
from sqlalchemy.orm import relationship,declarative_base,sessionmaker
from datetime import datetime
from contextlib import contextmanager
import bcrypt
import getpass
import urllib.parse
import os
from dotenv import load_dotenv
load_dotenv()


Base=declarative_base()

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(50), nullable=False, unique=True)
    __password   = Column("password", String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    phone      = Column(String(20), nullable=False, unique=True)
    # 建立与对话记录的关系（方便后续通过 user.conversations 直接获取）
    conversations = relationship("Conversation", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
    
    #这里打印phone时候需要隐藏中间的数字，只显示前三位和后四位
    def __str__(self):
        # 隐藏phone中间的数字
        if self.phone and len(self.phone) >= 7:
            masked_phone = f"{self.phone[:3]}****{self.phone[-4:]}"
        return f"User(id={self.id}, username='{self.username}', phone='{masked_phone}', created_at='{self.created_at}')"

    def __init__(self, username, phone):
        self.username = username
        self.phone = phone

    #这里需要给password加上@property装饰器，防止直接访问password属性，这个需要修改上面的password属性为_password,
    #并在__init__中将password赋值给_password   
    @property
    def password(self):
        raise AttributeError("密码字段不允许直接读取")

    @password.setter
    def password(self, value):    
        self.__password = self._hash_password(value)

    def _hash_password(self, password):
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    

class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role       = Column(Enum("user", "assistant"), nullable=False)
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    user = relationship("User", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(id={self.id}, role='{self.role}')>"
    
    def __str__(self):
        return f"Conversation(id={self.id}, user_id={self.user_id}, role='{self.role}', content='{self.content}', created_at='{self.created_at}')"

class MysqlDB:
    def __init__(self):
        password = urllib.parse.quote_plus(os.getenv('password'))
        db_url = f"mysql+pymysql://{os.getenv('user')}:{password
}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}?charset={os.getenv('charset')}"
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def __enter__(self):
        self.session = self.Session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
            print(f"An error occurred: {exc_val}")
        else:
            self.session.commit()
        self.session.close()
    

# @contextmanager
# def get_db():
#     password = urllib.parse.quote_plus(os.getenv('password'))
#     db_url = f"mysql+pymysql://{os.getenv('user')}:{password}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}?charset={os.getenv('charset')}"
#     print(db_url)
#     engine = create_engine(db_url)
#     Session = sessionmaker(bind=engine)
#     Base.metadata.create_all(engine)
#     db=SessionLocal()
#     try:
#         yield db
#         db.commit()
#     except Exception as e:
#         db.rollback()
#         raise e
#     finally:
#         db.close()

if __name__ == "__main__":
    #读取.env文件里的数据库配置
    #我的密码中含有@符号需要特殊处理，使用urllib.parse.quote_plus()进行编码
    with MysqlDB() as db:
        # 创建一个新用户
        
        print("开始注册新用户")
        input_username = input("请输入用户名: ")
        #输入秘密这里需要掩盖
        input_password = getpass.getpass("请输入密码: ")
        input_ag_password = getpass.getpass("请再次输入密码: ")
        if input_password != input_ag_password:
            print("两次输入的密码不一致，请重新运行程序。")
            exit(1)
        input_phone = input("请输入手机号: ")
        new_user = User(username=input_username, phone=input_phone)
        new_user.password = input_password  # 设置密码时会自动进行哈希处理
        db.add(new_user)
        db.commit()
        print(f"Created new user: {new_user}")

        # 创建一个新对话记录
        new_conversation = Conversation(user_id=new_user.id, role="user", content="Hello, this is a test conversation.")
        db.add(new_conversation)
        db.commit()
        print(f"Created new conversation: {new_conversation}")

        # 用户登陆
        print("开始用户登录")
        login_username = input("请输入用户名: ")
        login_password = getpass.getpass("请输入密码: ")
        user = db.query(User).filter(User.username == login_username).first()
        if user and user.verify_password(login_password, user._User__password):
            print(f"登录成功，欢迎 {user.username}!")
        else:
            print("用户名或密码错误，请重新运行程序。")
            exit(1)
        # 查询用户及其对话记录
        user_with_conversations = db.query(User).filter(User.id == new_user.id).first()
        print(f"User: {user_with_conversations}")
        for conv in user_with_conversations.conversations:
            print(f"Conversation: {conv}")
        

