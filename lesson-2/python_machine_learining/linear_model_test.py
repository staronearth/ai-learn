from sklearn.linear_model import LinearRegression,SGDRegressor,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def student_score_linear():
    #加载数据
    X = np.array([[80, 86], [82, 80], [85, 78], [90, 90],
     [86, 82], [82, 90], [78, 80], [92, 94]])
    y = np.array([84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4])
    #获取训练数据集和测试数据集
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=22)

    #特征提取进行数据归一化处理
    # scaler=StandardScaler()
    # X_train=scaler.fit_transform(X_train)
    # X_test=scaler.transform(X_test)

    #训练模型
    model=LinearRegression()
    model.fit(X_train,y_train)
    # model = SGDRegressor()
    # model.fit(X_train, y_train)
    #查看训练后的结构
    print("w的系数为:",model.coef_)
    print("b为:",model.intercept_)

    #预测新的样本点
    print("新样本[100,80]的预测为：",model.predict([[100,80]]))

def normal_equation():
    X = np.array([[80, 86], [82, 80], [85, 78], [90, 90],
     [86, 82], [82, 90], [78, 80], [92, 94]])
    y = np.array([84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4])

    w=np.linalg.inv(X.T@X)@X.T@y

    print("w为:",w)
    print(w@np.array([100,80]))

def gradient_desent():
    # 模拟数据
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    # 初始化参数 （随机值）
    w, b, alpha = 0, 0, 0.3
    # 迭代更新参数
    for i in range(150):
        # 预测值
        y_pred = w * x + b
        # 计算损失（均方误差）
        loss = (1/3) * np.sum((y_pred - y) ** 2)
        print(f"迭代 {i+1}: loss={loss:.4f}, w={w:.2f}, b={b:.2f}")
        # 计算梯度（求导）
        dw = (1/3) * np.sum((y_pred - y) * x)
        db = (1/3) * np.sum(y_pred - y)
        # 更新参数
        w -= alpha * dw
        b -= alpha * db
    print(f"w={w:.2f}, b={b:.2f}")

def linear_wine():
    
    # 加载数据集并划分训练集和测试集
    data = load_wine()
    print("样本点数量:", data.data.shape[0])
    print("特征名称:", data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=22)

    # 特征缩放（标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 模型1：正规方程
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    print("正规方程系数 w:", model_lr.coef_)
    print("正规方程截距 b:", model_lr.intercept_)

    # 模型2：梯度下降
    model_sgd = SGDRegressor(max_iter=2000, eta0=0.01)
    model_sgd.fit(X_train, y_train)
    print("SGD系数 w:", model_sgd.coef_)
    print("SGD截距 b:", model_sgd.intercept_)

    # 评估模型性能（均方误差）
    y_pred_lr = model_lr.predict(X_test)
    print("MSE (正规方程):", mean_squared_error(y_test, y_pred_lr))
    y_pred_sgd = model_sgd.predict(X_test)
    print("MSE (SGD):", mean_squared_error(y_test, y_pred_sgd))
    joblib.dump(model_sgd, 'wine_model.pkl')

def ridge_lasso():
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    import numpy as np

    np.random.seed(42)
    X = np.random.rand(100, 10)
    print(X.shape)
    y = X.dot([1,2,3,0,0,0,0,0,0,0]) + np.random.normal(0, 0.1, 100)

    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.1).fit(X, y)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)

    print("Ridge 系数:", ridge.coef_)
    print("Lasso 系数:", lasso.coef_)
    print("ElasticNet 系数:", elastic.coef_)

def load_model():
    # 加载数据集并划分训练集和测试集
    data = load_wine()
    print("样本点数量:", data.data.shape[0])
    print("特征名称:", data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=22)

    # 特征缩放（标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 加载
    loaded_model = joblib.load('wine_model.pkl')

    # 评估
    y_pred_sgd = loaded_model.predict(X_test)
    print("MSE (SGD):", mean_squared_error(y_test, y_pred_sgd))

def homework_1():
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2.2, 3.9, 5.8, 7.9, 10.1])

    # 正规方程解
    X_b = np.c_[X, np.ones_like(X)]  # 添加偏置列
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    print(f"系数 w: {w[0]:.2f}, 偏置 b: {w[1]:.2f}")

    plt.scatter(X, y)
    plt.plot(X, X_b @ w, 'r-')
    plt.show()

def homework_2():
    data = load_wine()
    print("样本点数量:", data.data.shape[0])
    print("特征名称:", data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=22)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 定义 SGD 分类器
    # loss='hinge' 是线性 SVM，也可以改成 'log_loss' 做逻辑回归
    model =  SGDClassifier(loss='hinge', max_iter=2000, random_state=42)
    # 设置要搜索的超参数网格
    # 常见的参数：alpha（正则化强度）、eta0（初始学习率）、learning_rate（学习率策略）
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'eta0': [0.01, 0.05, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("最佳 C:", grid.best_params_)
    print("最佳交叉验证得分:", grid.best_score_)
    # 在测试集上评估
    test_score = grid.score(X_test, y_test)
    print("测试集准确率: {:.3f}".format(test_score))
    
    y_pred_sgd = grid.best_estimator_.predict(X_test)
    print("测试集准确率 (SGD): {:.3f}".format(accuracy_score(y_test, y_pred_sgd)))
    joblib.dump(grid.best_estimator_, 'wine_sgd_model.pkl')
def homework_3():
    data = load_wine()
    print("样本点数量:", data.data.shape[0])
    print("特征名称:", data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=22)

    # 特征缩放（标准化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    sample_scaled = scaler.transform([[13.2, 2.77, 2.51, 18.5, 96.0, 1.8, 0.85, 3.3, 1.2, 5.7, 1.0, 3.1, 1060]])

    model = joblib.load('wine_sgd_model.pkl')
    pred = model.predict(sample_scaled)
    print("预测质量:", pred)

if __name__=="__main__":
    # student_score_linear()
    # normal_equation()
    # gradient_desent()
    # linear_wine()
    # ridge_lasso()
    # load_model()
    # homework_1()
    homework_2()
    # homework_3()
