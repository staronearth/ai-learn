import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
from sqlalchemy import create_engine
import urllib
import os
from dotenv import load_dotenv
load_dotenv()

def create_dataframe():
    #字典创建
    df1=pd.DataFrame({
        'name':['bob','alex','rose'],
        'age':[18,19,20],
        '成绩':[92,99,100]
    })
    print(df1)

    #嵌套列表+colume
    df2=pd.DataFrame([["bob",18,92],["alex",19,99],["rose",20,100]],columns=["name","age","成绩"])
    print(df2)

def dataframe_poperty():
    df1=pd.DataFrame({
        'name':['bob','alex','rose'],
        'age':[18,19,20],
        '成绩':[92,99,100]
    })
    print(df1.shape)
    print(df1.index)
    print(df1.columns)
    print(df1.dtypes)
    print(df1.info())
    print(df1.describe())

def get_dataframe():
    df1=pd.DataFrame({
        'name':['bob','alex','rose'],
        'age':[18,19,20],
        '成绩':[92,99,100]
    },index=['h1','h2','h3'])
    print(df1['age'])
    print(df1[['name','成绩']])
    print(df1[df1['成绩']>90])
    print(df1.loc['h1','成绩'])
    print(df1.iloc[0,2])
    print(df1.loc[df1['成绩']>90,"name"])

def dataframe_colume():
    df1=pd.DataFrame({
        'name':['bob','alex','rose'],
        'age':[18,19,20],
        '成绩':[92,99,100]
    },index=['h1','h2','h3'])
    df1["是否及格"]=df1['成绩']>90

    print(df1)

    df1.drop(columns=["是否及格"],inplace=True)
    print(df1)

def queshi_value():
    df1=pd.DataFrame({
        'name':['bob','alex','rose'],
        'age':[18,np.nan,20],
        '成绩':[92,99,100]
    },index=['h1','h2','h3'])
    print(df1.isnull().sum())
    print(df1.dropna())
    print(df1.fillna(20))

def dataframe_groupby():
    df = pd.DataFrame({
        '班级': ['A', 'B', 'A', 'A', 'B'],
        '成绩': [85, 76, 92, 88, 69]
    })
    gp_df=df.groupby('班级').agg(
        总成绩=('成绩','sum'),
        平均成绩=('成绩','mean'),
        总的个数=('成绩','count')
    )
    print(gp_df)

def merge_dataframe():
    df1 = pd.DataFrame({'ID': [1,2,3], '姓名': ['张三','李四','王五']})
    df2 = pd.DataFrame({'ID': [2,3,4], '成绩': [85,90,78]})

    inner = pd.merge(df1, df2, on='ID', how='inner').ffill()    # 内连接，只保留两表共有的 ID 2 和 3
    print(inner)
    left  = pd.merge(df1, df2, on='ID', how='left').bfill()    # 左连接，保留 df1 的所有行，df2 中没有匹配的 ID 1 会填充 NaN
    print(left)
    outer = pd.merge(df1, df2, on='ID', how='outer').bfill()    # 全外连接，保留两表所有行，ID 1 和 4 会填充 NaN
    print(outer)
    print(pd.concat([df1, df2], axis=0))   # 按行堆叠
    print(pd.concat([df1, df2], axis=1))   # 按列拼接

def dataframe_export():
    # CSV文件的导出 与 读取
    df = pd.DataFrame({
        '班级': ['A', 'B', 'A', 'A', 'B'],
        '成绩': [85, 76, 92, 88, 69]
    }, index=['h1', 'h2', 'h3', 'h4', 'h5'])
    print(df)
    df.to_csv('output.csv', index=False)   # 保存为 CSV 文件，包含索引
    iris_df = pd.read_csv('iris.csv', encoding='utf-8')
    print(iris_df.head())

    # Excel（多个 Sheet）
    df1 = pd.DataFrame({'ID': [1,2,3], '姓名': ['张三','李四','王五']})
    df2 = pd.DataFrame({'ID': [2,3,4], '成绩': [85,90,78]})
    with pd.ExcelWriter('output.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)

    # 读取 Excel 某 sheet
    df1 = pd.read_excel('output.xlsx', sheet_name='Sheet1')
    print(df1)

def dataframe_show():
    df = pd.DataFrame({
        '班级': ['A', 'B', 'A', 'A', 'B'],
        '成绩': [85, 76, 92, 88, 69]
    }, index=['h1', 'h2', 'h3', 'h4', 'h5'])
    # 全局设置
    plt.rcParams['font.sans-serif'] = ['SimHei']      # 显示中文
    plt.rcParams['axes.unicode_minus'] = False        # 显示负号

    df['成绩'].plot(kind='line', title='score', grid=True)
    plt.show()

def homwork_1():
    print(matplotlib.get_cachedir())
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 读取与清洗
    df = pd.read_excel('source.xlsx').fillna(0)

    # 2. 计算总分与及格判定
    df['总分'] = df['平时成绩'] * 0.3 + df['考试成绩'] * 0.7
    df['是否及格'] = df['总分'].apply(lambda x: '及格' if x >= 60 else '不及格')

    # 3. 统计及格率
    pass_rate = (df['是否及格'] == '及格').mean()
    print(f"及格率: {pass_rate:.2%}")

    # 4. 饼图
    counts = df['是否及格'].value_counts()
    counts.plot(kind='pie', autopct='%1.1f%%', explode=(0, 0.05))
    plt.title('及格率分布')
    plt.savefig('pass_rate.png')
    plt.show()

    # 5. 导出
    df.to_excel('处理后的成绩表.xlsx', index=False)

def homework_2():
    # 读取 UCI Iris 数据集
    # 替换表头为指定的names
    df = pd.read_csv('iris.csv',names=['花瓣长', '花瓣宽', '花萼长', '花萼宽', '种类'], header=0)

    print(df.info())
    print(df.describe())

    # 分组统计：每种花的平均花萼长度
    print(df.groupby('种类')['花萼长'].mean())

    # 箱线图：花瓣宽度按种类分布
    df.boxplot(column='花瓣宽', by='种类')
    plt.title('不同种类的花瓣宽度分布')
    plt.suptitle('')
    plt.show()

    # 交叉表：花瓣宽度分箱与种类的关系
    cross = pd.crosstab(
        index=pd.cut(df['花瓣宽'], bins=3),
        columns=df['种类'],
        margins=True
    )
    print(cross)

def homework_3():
    # 生成 15 天日期序列
    dates = pd.date_range(start='2025-04-26', periods=15, freq='D')

    # 模拟销售额
    np.random.seed(0)
    sales = pd.Series(np.random.randint(10, 100, 15), index=dates)

    # 滚动 7 日求和 vs 按周重采样
    print(sales.rolling(window=7).sum())   # 每天显示前 7 天累计
    print(sales.resample('W').sum())       # 每周合计（W-SUN）

def csv_to_mysql():   
    password = urllib.parse.quote_plus(os.getenv('password'))
    db_url = f"mysql+pymysql://{os.getenv('user')}:{password
    }@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}?charset={os.getenv('charset')}"
    # 读取 CSV
    df = pd.read_csv('iris.csv')

    # 创建数据库引擎（pymysql 作为底层驱动）
    engine = create_engine(db_url)

    # 写入表（if_exists='replace' 表示重建表）
    df.to_sql('interviews', engine, if_exists='replace', index=False)
    print("数据写入成功")

def mysql_to_csv():
    password = urllib.parse.quote_plus(os.getenv('password'))
    db_url = f"mysql+pymysql://{os.getenv('user')}:{password
    }@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}?charset={os.getenv('charset')}"
    engine = create_engine(db_url)
    # 方式1：使用 read_sql_query 直接读取
    df = pd.read_sql_query("SELECT * FROM interviews WHERE species='setosa'", engine)
    print(df.head())

    # 方式2：使用 read_sql_table 读取整表
    df_all = pd.read_sql_table('interviews', engine)
    print(df_all.tail())
    df.to_csv("mysql_csv.csv")

def homework_4():
    #创建包含“产品”、“单价”、“销量”三列的 DataFrame，新增“总金额=单价×销量”列，筛选总金额 > 1000 的行。
    df=pd.DataFrame({
        '产品': ['A', 'B', 'C'],
        '单价': [100, 200, 150],
        '销量': [8, 6, 12]
    }
    )
    df["总金额"]=df["单价"]*df["销量"]
    print(df)
    print(df[df['总金额']>1000])

def homework_5():
    #按“部门”分组，分别统计每个部门的平均薪资和人数，结果按平均薪资降序排列。
    df = pd.DataFrame({
        '部门': ['技术', '销售', '技术', '人事', '销售'],
        '薪资': [15000, 8000, 20000, 6000, 9000]
    })

    new_df=df.groupby('部门').agg(
        平均薪资=("薪资",'mean'),
        人数=("薪资",'count')
    )
    print(new_df)

def homework_6():
    #读取一个 CSV 文件，检查并填充缺失值（数值列填 0，字符串列填"未知"），删除完全重复的行，输出清洗后的形状
    df=pd.read_csv('output.csv')
    print(df)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col]=df[col].fillna("未知")
        else:
            df[col]=df[col].fillna(0)
    print(df)

def homework_7():
    dates = pd.date_range('2026-07-01', '2026-07-31', freq='D')
    visitors = pd.Series(np.random.randint(100, 500, len(dates)), index=dates)
    weekly = visitors.resample('W').sum()
    weekly.plot(kind='line', title='每周访客量', marker='o')
    plt.show()

if __name__=="__main__":
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    # create_dataframe()
    # dataframe_poperty()
    # get_dataframe()
    # dataframe_colume()
    # queshi_value()
    # dataframe_groupby()
    # merge_dataframe()
    # dataframe_export()
    # dataframe_show()
    # homwork_1()
    # homework_2()
    # homework_3()
    # csv_to_mysql()
    # mysql_to_csv()
    # homework_4()
    # homework_5()
    # homework_6()
    homework_7()