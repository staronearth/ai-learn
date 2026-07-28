import numpy as np
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start)*1000  # Convert to milliseconds
        print(f"Function {func.__name__} took {duration}ms to execute.")
        return result
    return wrapper

@timer
def list_to_square(list):
    return [x**2 for x in list]

@timer
def array_to_square(array):
    return array**2


def list_ndarry_performance(size):
    my_list = list(range(size))
    my_array = np.array(my_list)

    list_result = list_to_square(my_list)
    array_result = array_to_square(my_array)

def ndarry_craete():
    a1=np.array(1)
    print(a1)
    #元组创建,元组可以，异构但是list必须存储相同的数据结构
    a2=np.array((1,2,3,4.0))
    print(a2)
    a3=np.array([1,2,3,4])
    print(a3)

    #制定数据类型
    a4=np.array([1,2,3,4],dtype=np.float32)
    print(a4)
    a5=np.array([1,2,3,4],dtype=np.uint8)
    print(a5)

def struct_ndarry():
    stu_type=np.dtype([('name','U10'), ('age', 'i4'), ('score', 'f4')])
    abstudent=np.array([('Tom', 20, 88.5), ('Jerry', 19, 92.0)], dtype=stu_type)
    print(abstudent)

def special_ndarry():
    #创建全0数组
    a1=np.zeros((2,3))
    print(a1)
    #创建全1数组
    a2=np.ones((2,3))
    print(a2)
    #创建单位矩阵
    a3=np.eye(3)
    print(a3)
    #创建等差数列
    a4=np.arange(1,10,2)
    print(a4)
    #创建等差数列，指定数据类型
    a5=np.arange(1,10,2,dtype=np.float32)
    print(a5)
    #创建等差
    a6=np.linspace(1,10,5)
    print(a6)
    a7=np.logspace(0,3,3)
    print(a7)
    #创建空数组
    print(np.empty((2,3)))

def ndim_array():
    #一维数组
    a1=np.array([1,2,3,4])
    print(a1)
    #二维数组
    a2=np.array([[1,2,3],[4,5,6]])
    print(a2)
    #三维数组
    a3=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
    print(a3.shape)
    #四维数组
    a4=np.zeros((2,3,4,5))
    print(a4.shape)
    #五维数组
    a5=np.ones((2,3,4,5,6),dtype=np.float64)
    print(a5.shape,a5.ndim,a5.size,a5.dtype,a5.itemsize,a5.nbytes)

def ndarray_slice():
    #一维数组
    a1=np.array([1,2,3,4,5])
    print(a1[0],a1[-1])
    print(a1[0:3])
    print(a1[::-1])
    #二维数组
    a2=np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(a2[2,2])
    print(a2[0:1,:])
    print(a2[:,0])
    print(a2[:,[0,2]])

    #布尔索引
    print(a2[a2>5])

    a2[0,0]=10
    print(a2)

def change_ndarrays():
    #数据类型转换
    a1= np.array([1.1,2.3,3.9])
    print(a1.dtype)
    a2=a1.astype(np.int32)
    print(a2)
    #形状变换
    a3=np.arange(1,13)
    print(a3.shape)
    print(a3)

    #改变形状
    a4=a3.reshape((3,4)) #副本
    print(a4.shape)
    print(a4)
    print(a3)

    a3.resize((3,4)) # 原地修改形状 (就地修改)
    print(a3.shape)
    print(a3)

    a5=a3.flatten()
    a5[1]=9
    print(a5)
    print(a3)

    a6=a3.ravel()
    a6[1]=99
    print(a6)
    print(a3)

    #转置
    print(a3.T)
    a7=np.swapaxes(a3.T,axis1=0,axis2=1)
    print(a7)

def expend_ndarrys():
    a1=np.array([1,2,3])
    print(a1.shape)

    #增加纬度
    a2=np.expand_dims(a1,axis=0)
    print(a2.shape)
    print(a2)
    print(a2[0,1])

    a3=np.expand_dims(a1,axis=1)
    print(a3.shape)
    print(a3)
    print(a3[2,0])
    print(a1)

    a4=a3.squeeze(axis=1)
    print(a4)    

    a5=a2.squeeze(axis=0)
    print(a5)

def math_statistics():
    np.random.seed(42)                # 固定种子（结果复现）

    # 均匀分布 [0,1)
    print(np.random.rand(3, 2))       # 形状 (3,2)
    # 标准正态分布 (均值0，标准差1)
    print(np.random.randn(1000).mean())  # 接近 0
    # 指定正态分布
    print(np.random.normal(loc=1, scale=3, size=1000).std())  # 接近 3
    # 随机整数
    print(np.random.randint(10, 20, size=(2,3)))
    # 打乱数组
    data = np.arange(10)
    np.random.shuffle(data)
    print(data)

    arr = np.array([[4, 2, 9],
                [1, 8, 6],
                [7, 5, 3],
                [1, 3, 5]])

    print(np.mean(arr))        # 4.5
    print(np.max(arr))         # 9
    print(np.min(arr))         # 1
    print(np.var(arr))         # 方差
    print(np.std(arr))         # 标准差
    print(np.median(arr))      # 中位数

    # 指定轴统计 a.shape = (4, 3)
    print(arr.max(axis=0))     # 每列最大值 [7 8 9]
    print(arr.max(axis=1))     # 每行最大值 [9 8 7 5]
    # 轴向统计后该轴维度消失
    print(np.argmax(arr))           # 全局最大值的扁平索引 -> 8
    print(np.argmax(arr, axis=1))   # 每行最大值的列索引 -> [2 1 0 2]
    # 布尔索引直接获取满足条件的元素
    print(arr[arr > 5])         # [9 8 6 7]
    # np.where 三种用法
    idx = np.where(arr > 5)     # 返回满足条件的索引元组 
    print(idx) # (array([0, 1, 1, 2]), array([2, 1, 2, 0])) (0,2) (2,1) (1,2) (2,0)
    result = np.where(arr > 5, arr, 0)  # 满足保留，否则置 0
    print(result)
    result = np.where(arr > 5, '高', '低') # 三元模式
    print(result)

    sorted_arr = np.sort(arr, axis=0)        # 沿列排序，返回副本
    print(sorted_arr)
    print(arr)
    arr.sort(axis=0)                         # 原地排序
    print(arr)
    idx = np.argsort(arr[:, 0])              # 返回第一列排序索引
    print(idx)

def ndarry_cost():
    '''数组的广播机制 
    1.纬度从右向左，如果纬度相同可以直接相加
    3.如果纬度没有就设为1,如果纬度为1就广播,其中一个维度是 1（NumPy 会“拉伸”这个维度来匹配对方）；
    '''
    a1=np.array([[1,2,3],[4,5,6]])
    a2=np.array([10,2,30])
    print(a1+a2)

    a=np.ones((2,3))
    b=np.ones((3,1))
    print(a+b)

def ndarry_stack_and_split():
    a = np.array([[1, 2], [3, 4]])   # (2,2)
    b = np.array([[5, 6]])           # (1,2)

    # 垂直堆叠
    print(np.vstack((a, b)))                # (3,2)
    # 水平堆叠
    print(np.hstack((a, b.T)))              # (2,3)
    # 按列堆叠（一维数组会转为列）
    print(np.column_stack((a, b.T)))    
    # 沿指定轴连接
    print(np.concatenate((a, b), axis=0))
    # 沿新轴堆叠（升维）
    print(np.stack((a, a)))                 # (2,2,2)
    print(np.stack((b, b)))                 # (2,1,2) 

    arr = np.array([1,2,3,4,5,6])
    print(np.split(arr, 2))                 # 均分为 2 份
    print(np.split(arr, [2, 4]))            # 在索引 2 和 4 处分割
    print(np.hsplit(arr.reshape(2,3), [2])) # 水平分割
    print(np.array_split(arr, 4))           # 不等分

def ndarr_rand_num():
    np.random.seed(42)
    print(np.random.rand(3,2))
    # 标准正态分布 (均值0，标准差1)
    print(np.random.randn(100).mean())
    # 指定正态分布
    print(np.random.normal(loc=1,scale=3,size=1000).std())
    # 随机整数
    print(np.random.randint(10, 20, size=(2,3)))
    # 打乱数组
    data = np.arange(10)
    np.random.shuffle(data)
    print(data)

def numpy_math_num():
    arr = np.array([[4, 2, 9],
                [1, 8, 6],
                [7, 6, 3],
                [1, 3, 7]])

    print(np.mean(arr))        # 4.5
    print(np.max(arr))         # 9
    print(np.min(arr))         # 1
    print(np.var(arr))         # 方差
    print(np.std(arr))         # 标准差
    print(np.median(arr))      # 中位数

    # 指定轴统计 a.shape = (4, 3)
    print(arr.max(axis=0))     # 每列最大值 [7 8 9]
    print(arr.max(axis=1))     # 每行最大值 [9 8 7 5]
    # 轴向统计后该轴维度消失
    print(np.argmax(arr))           # 全局最大值的扁平索引 -> 82
    print(np.argmax(arr, axis=1))   # 每行最大值的列索引 -> [2 1 0 2]

    # 布尔索引直接获取满足条件的元素
    print(arr[arr > 5])         # [6 7 8 9]

    # np.where 三种用法
    idx = np.where(arr > 5)     # 返回满足条件的索引元组
    print(idx)
    result = np.where(arr > 5, arr, 0)  # 满足保留，否则置 0
    print(result)
    result = np.where(arr > 5, '高', '低') # 三元模式
    print(result)

    sorted_arr = np.sort(arr, axis=0)        # 沿列排序，返回副本
    print(sorted_arr)
    arr.sort(axis=0)                         # 原地排序
    print(arr)
    idx = np.argsort(arr[:, 0])              # 返回第一列排序索引
    print(idx)
def ndarray_conn_and_split():
    a = np.array([[1, 2], [3, 4]])   # (2,2)
    b = np.array([[5, 6]])           # (1,2)

    # 垂直堆叠
    print(np.vstack((a, b)))                # (3,2)
    # 水平堆叠
    print(np.hstack((a, b.T)))              # (2,3)
    # 按列堆叠（一维数组会转为列）
    print(np.column_stack((a, b.T)))    
    # 沿指定轴连接
    print(np.concatenate((a, b), axis=0))
    # 沿新轴堆叠（升维）
    print(np.stack((a, a)))                 # (2,2,2)
    print(np.stack((b, b)))                 # (2,1,2) 
    arr = np.array([1,2,3,4,5,6])
    print(np.split(arr, 2))                 # 均分为 2 份
    print(np.split(arr, [2, 4]))            # 在索引 2 和 4 处分割
    print(np.hsplit(arr.reshape(2,3), [2])) # 水平分割
    print(np.array_split(arr, 4))           # 不等分

def ndarray_inner():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # 矩阵乘法
    print(A @ B)
    print(np.dot(A, B))

    # 内积（一维向量点积，多维按最后轴计算）
    print(np.inner([1,2], [3,4]))    # 11

    # 外积（向量转置相乘）
    print(np.outer([1,2,3], [6,8]))    # 

    # 逆矩阵
    print(np.linalg.inv(A))

    # 行列式
    print(np.linalg.det(A))

    # 特征值与特征向量
    vals, vecs = np.linalg.eig(A)
    print(vals, vecs)

def execise_1():
    #生成 30 名学生的 3 科成绩（随机整数 40~100），
    # 计算每位学生的总分、平均分，找出总分前 5 名，统计每科平均分和最高分，筛选出单科不及格（<60）的学生并标记。
    np.random.seed(1)

    score=np.random.randint(40,100,size=(30,3))
    print(score)

    print(score.sum(axis=1))
    print(np.mean(score,axis=1))

    print(np.sort(score.sum(axis=1))[::-1][:5])

    print(score.mean(axis=0))
    print(score.max(axis=0))
    # print(np.)
    print(np.where(score>60,score,0))

    # 计算学生之间的成绩差异（标准差）
    std_per_student = score.std(axis=1)
    print("成绩波动最大的学生索引:", np.argmax(std_per_student))

def homework_1():
    #创建一个 5×5 的单位矩阵，并将对角线上移一位的元素替换为 1（例如生成一个简单的双对角线矩阵）。
    arr=np.eye(5,k=0)+np.eye(5,k=1)+np.eye(5,k=-1)
    print(arr)
    #随机生成 100 个服从正态分布（均值 70，标准差 10）的学生分数，统计分数在 60 以下和 90 以上的比例。
    np.random.seed(77)
    socre=np.random.normal(loc=70,scale=10,size=100)

    low=np.mean(socre<60)
    high=np.mean(socre>90)
    print(low,high)
    print(socre)
    #利用广播计算两个矩阵的距离：给定 5 个点的坐标 (5,2) 和 3 个点的坐标 (3,2)，
    # 求每个点到其他点的欧氏距离（结果形状 (5,3)）
    p1 = np.random.rand(5, 2)
    p2 = np.random.rand(3, 2)
    # 利用广播 (5,1,2) - (1,3,2) -> (5,3,2)
    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    print(dist)   # (5,3)
if __name__=="__main__":
    #np随机数和正太分布
    #ndarr_rand_num()
    # numpy_math_num()
    #矩阵的拼接与分割
    # ndarray_conn_and_split()
    #矩阵的运算
    # ndarray_inner()
    # ndarry_stack_and_split()
    #对比列表和ndarray的性能
    # list_ndarry_performance(1000000)
    # ndarry_craete()
    # special_ndarry()
    # struct_ndarry()
    # ndim_array()
    # ndarray_slice()

    # change_ndarrays()

    # expend_ndarrys()
    # ndarry_cost()
    # math_statistics()
    #课后练习一
    # execise_1()
    #homework1
    homework_1()
