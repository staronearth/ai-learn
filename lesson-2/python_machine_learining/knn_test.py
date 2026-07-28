from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def diff_distance(weight):
    def dist(x,y):
        diff=(x-y)**2
        return np.sqrt(np.sum(weight*diff))
    return dist

#自定义距离：余弦距离（1 - 余弦相似度）
def cosine_distance(x, y):
    dot = np.dot(x, y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return 1 - dot / (norm + 1e-8)  # 加小量防止除0
def movie_knn():
    from sklearn.neighbors import KNeighborsClassifier

    # 1. 准备数据（特征 X 和 标签 y）
    # 特征：搞笑镜头, 拥抱镜头, 打斗镜头
    X = [
        [39, 0, 31], [3, 2, 65], [2, 3, 55], [9, 38, 2],
        [8, 34, 17], [5, 2, 57], [21, 17, 5], [45, 2, 9]
    ]
    # 标签
    y = ['喜剧片', '动作片', '爱情片', '爱情片',
        '爱情片', '动作片', '喜剧片', '喜剧片']

    w=np.array([2.0,0.5,1.0])
    # 2. 创建模型并训练
    # model = KNeighborsClassifier(n_neighbors=3)
    #这个是函数自带的使用距离作为权重
    # model = KNeighborsClassifier(n_neighbors=3,weights="distance")  # K值取3
    model=KNeighborsClassifier(n_neighbors=5,metric=cosine_distance,algorithm='brute',weights='distance')
    model.fit(X, y)

    # 3. 预测新电影《唐人街探案》
    test = [[23, 3, 17]]  # 23个搞笑镜头，3个拥抱镜头，17个打斗镜头
    print(f"预测类型: {model.predict(test)[0]}")  # 输出：喜剧片

    # 4. 评估准确率
    accuracy = model.score(X, y)  # 在训练集上评估（实际应用应在测试集上）
    print(f"准确率: {accuracy:.2%}")

def iris_data_analysis():
    # 获取鸢尾花并查看特征分布
    import pandas as pd
    import matplotlib.pyplot as plt
    from  sklearn.datasets import load_iris

    iris = load_iris()

    iris_df = pd.DataFrame(iris["data"], columns=["花瓣长", "花瓣宽", "花萼长", "花萼宽"])
    iris_df["类别"] = iris.target
    print(iris_df.head())

    # 绘制散点图1：看一下 花萼长 VS 花瓣长
    plt.figure()
    colors = ["red", "green", "blue"]
    for species in range(3):
        subset = iris_df[iris_df["类别"] == species]
        plt.scatter(subset["花萼长"], subset["花瓣长"], c=colors[species], label=f"Species {species}")
    plt.xlabel("花萼长")
    plt.ylabel("花瓣长")
    plt.legend()
    plt.show()

    # 绘制散点图2：看一下 花萼长 VS 花瓣宽
    plt.figure()
    colors = ["red", "green", "blue"]
    for species in range(3):
        subset = iris_df[iris_df["类别"] == species]
        plt.scatter(subset["花萼长"], subset["花瓣宽"], c=colors[species], label=f"Species {species}")
    plt.xlabel("花萼长")
    plt.ylabel("花瓣宽")
    plt.legend()
    plt.show()

def iris_knn():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer
    from sklearn.pipeline import Pipeline
    # 1. 获取数据
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. 划分训练集和测试集 比例 8:2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=22
    )

    # 3. 【关键步骤】特征预处理：标准化
    # 让所有特征都变成均值为0、标准差为1的分布，消除量纲影响
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # 在训练集上计算均值和标准差，并转换
    X_test = scaler.transform(X_test)         # 使用和训练集一样的参数转换测试集

    # 4. 模型训练 + 网格搜索（自动找最佳K值）
    # param_grid = {"n_neighbors": [1,2, 3,4,5,6, 7, 9,11]}
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(weights='distance'))
    ])
    param_grid = [
        {
            'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
            'knn__n_neighbors': range(3, 11),
            'knn__metric': ['euclidean', 'manhattan', 'cosine']
        },
        {
            'scaler': [Normalizer()],  # Normalizer 常配合余弦距离
            'knn__n_neighbors': range(3, 11),
            'knn__metric': ['cosine']
        }
    ]
    model = KNeighborsClassifier(weights='distance')
    # 什么是交叉验证？就是将拿到的训练数据，分为训练和验证集。
    # 5折交叉验证，就是将数据分为5份，其中一份作为验证集，然后经过5次测试，每次都更换不同的验证集。
    # 就得到5组模型的结果，取平均值作为最终结果。
    grid_search = GridSearchCV(pipe, param_grid, cv=5)  # 使用5折交叉验证
    grid_search.fit(X_train, y_train)

    # 5. 模型评估
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(grid_search.best_params_)
    print(f"最佳K值: {grid_search.best_params_['knn__n_neighbors']}")
    print(f"测试准确率: {accuracy:.2%}")
    y_pred = best_model.predict(X_test)
    wrong_indices = np.where(y_pred != y_test)[0]

    print(f"错误样本数: {len(wrong_indices)}")
    for idx in wrong_indices:
        # idx 是 X_test 中的位置
        print(f"\n测试集索引: {idx}")
        print("特征值:", X_test[idx])
        print("真实标签:", iris.target_names[y_test[idx]])
        print("预测标签:", iris.target_names[y_pred[idx]])
        
        # 如果你还想看这个样本的 K 个邻居，可以继续
        distances, neighbors = best_model.named_steps['knn'].kneighbors([X_test[idx]])
        print("邻居索引:", neighbors[0])
        print("邻居标签:", y_train[neighbors[0]])
    # 6. 用最佳模型预测一个全新的样本
    sample = [[5.1, 3.5, 1.4, 0.2]]
    sample_scaled = scaler.transform(sample)
    prediction = best_model.predict(sample_scaled)
    print(f"预测品种: {iris.target_names[prediction[0]]}")

if __name__=="__main__":
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']      # 显示中文
    plt.rcParams['axes.unicode_minus'] = False        # 显示负号
    # movie_knn()
    # iris_data_analysis()
    iris_knn()