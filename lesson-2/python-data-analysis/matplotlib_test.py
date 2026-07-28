import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def first_pic():
    import matplotlib.pyplot as plt
    import numpy as np


    # 准备数据
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.sin(x)

    # 绘图
    plt.plot(x, y, 'r-o', label='sin(x)')

    # 美化/标注
    plt.title('正弦曲线')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)

    # 显示/保存
    plt.show()

def plot_pic():
    x = np.arange(0, 3 * np.pi, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, 'b-', label='sin', linewidth=2)
    plt.plot(x, y2, 'r--', label='cos', linewidth=2)
    plt.legend()
    plt.show()

    x = np.arange(0, 10)
    y = np.random.randint(1, 10, size=(10,))
    plt.plot(x, y, "r-o")
    plt.show()

    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.axis('equal')          # 等比例坐标轴
    plt.show()

def scatter_pic():
    np.random.seed(0)
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)          # 颜色映射
    sizes = 500 * np.random.rand(50)     # 点大小

    plt.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis')
    plt.colorbar()                       # 显示颜色条
    plt.show()

def bar_pic():
    classes = ['A班', 'B班', 'C班', 'D班']
    scores = [85, 92, 78, 88]

    plt.bar(classes, scores, color='steelblue', edgecolor='black')
    plt.ylabel('平均分')
    plt.show()
    #水平
    plt.barh(classes, scores, color='coral')
    plt.show()

def pie_pic():
    sizes = [30, 25, 20, 15, 10]
    labels = ['优秀', '良好', '中等', '及格', '不及格']
    explode = (0, 0, 0.1, 0, 0)      # 突出第三块

    #起始方向startangle=180度
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            startangle=180)
    plt.axis('equal')
    plt.show()

def hist_pic():
    data = np.random.randn(1000)      # 标准正态分布数据
    #bins：柱的个数（或自定义区间）。
    #density=True：归一化，纵轴表示概率密度，曲线下面积为 1。
    plt.hist(data, bins=100, density=True, alpha=0.7, color='skyblue')
    plt.xlabel('数值')
    plt.ylabel('频率密度')
    plt.show()

def boxplot_pic():
    # 模拟三个班级的成绩
    data = [np.random.normal(70, 10, 100),
            np.random.normal(75, 8, 100),
            np.random.normal(80, 12, 100)]

    plt.boxplot(data, tick_labels=['A班', 'B班', 'C班'], patch_artist=True)
    plt.ylabel('成绩')
    plt.show()

def sub_plot_pic():
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.subplot(2, 2, 1)      # 2行2列第1个
    plt.plot(x, y1)
    plt.subplot(2, 2, 2)
    plt.plot(x, y2)
    plt.show()

def subplots_pic():
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(x, y1)
    axes[0, 1].scatter(x, y2)
    # ...
    plt.tight_layout()          # 自动调整子图间距
    plt.show()

def plot_save_pic():
    data = [np.random.normal(70, 10, 100),
        np.random.normal(75, 8, 100),
        np.random.normal(80, 12, 100)]

    plt.boxplot(data, tick_labels=['A班', 'B班', 'C班'], patch_artist=True)
    plt.ylabel('成绩')
    plt.savefig('output.png', dpi=300, bbox_inches='tight')   # 高分辨率保存
    plt.show()

def plt_read_png():
    img = plt.imread('logo.png')        # 返回 numpy 数组，形状 (高, 宽, 通道)
    print(type(img))
    print(img.shape)
    plt.imshow(img)
    plt.axis('off')                     # 隐藏坐标轴
    plt.show()

def plt_oo_pic():
    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y2 = np.cos(x)

    # 绘图
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, y, 'r-')
    ax1.set_title('面向对象绘图')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x, y2)
    plt.show()

def plt_3d_pic():
    # 生成示例数据
    np.random.seed(42)          # 固定随机种子，保证可重复性
    n = 100                     # 点的数量
    x = np.random.rand(n) * 10  # X 坐标，范围 0~10
    y = np.random.rand(n) * 10  # Y 坐标
    z = np.random.rand(n) * 10  # Z 坐标
    colors = np.random.rand(n)  # 颜色值（数值，会自动映射到 colormap）
    sizes = np.random.rand(n) * 100  # 点的大小（面积，像素）

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=colors, s=sizes, cmap='viridis', alpha=0.8)


    # 添加颜色条
    plt.colorbar(sc, label='颜色值')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def student_socre_anlysis():
    # 模拟数据
    np.random.seed(42)
    df = pd.DataFrame({
        '班级': np.random.choice(['A班', 'B班', 'C班'], 150),
        '数学': np.random.normal(70, 10, 150).clip(0, 100),
        '语文': np.random.normal(68, 12, 150).clip(0, 100),
        '英语': np.random.normal(72, 11, 150).clip(0, 100)
    })

    # 总分列
    df['总分'] = df[['数学', '语文', '英语']].sum(axis=1)

    # 创建画布和子图
    fig = plt.figure(figsize=(14, 10))

    # 1. 各科成绩箱线图
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.boxplot([df['数学'], df['语文'], df['英语']], tick_labels=['数学', '语文', '英语'], patch_artist=True)
    ax1.set_title('各科成绩分布')
    ax1.set_ylabel('分数')

    # 2. 总分直方图
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(df['总分'], bins=20, color='lightgreen', edgecolor='black')
    ax2.set_title('总分分布')
    ax2.set_xlabel('总分')

    # 3. 班级平均分条形图
    ax3 = fig.add_subplot(2, 2, 3)
    mean_scores = df.groupby('班级')['总分'].mean()
    ax3.bar(mean_scores.index, mean_scores.values, color='skyblue', edgecolor='black')
    ax3.set_title('各班平均总分')
    ax3.set_ylabel('平均总分')

    # 4. 数学-语文散点图
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df['数学'], df['语文'], alpha=0.6, c='coral')
    ax4.set_title('数学 vs 语文')
    ax4.set_xlabel('数学成绩')
    ax4.set_ylabel('语文成绩')

    plt.tight_layout()
    plt.savefig('成绩分析报告.png', dpi=150, bbox_inches='tight')
    plt.show()

def homework_1():
    x = np.linspace(-5, 5, 20)
    y = x ** 2
    plt.plot(x, y, 'b--o', markerfacecolor='red')
    plt.title('二次函数 y = x^2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def homework_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(0, 2*np.pi, 100)
    ax1.plot(x, np.sin(x))
    ax1.set_title('sin(x)')
    ax2.plot(x, np.cos(x), 'r')
    ax2.set_title('cos(x)')
    plt.show()

def homework_3():
    df = pd.read_csv('iris.csv',names=['花萼长', '花萼宽', '花瓣长', '花瓣宽', '种类'], header=0)
    species_map = {'setosa':0, 'versicolor':1, 'virginica':2}
    df['类型'] = df['种类'].map(species_map)

    colors = df['类型']
    plt.scatter(df['花瓣长'], df['花瓣宽'], c=colors, cmap='Set1', alpha=0.7)
    plt.colorbar(ticks=[0,1,2], label='种类')
    plt.xlabel('花瓣长 (cm)')
    plt.ylabel('花瓣宽 (cm)')
    plt.show()

def homework_4():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.linspace(0, 10, 100)

    axes[0,0].plot(x, np.sin(x))
    axes[0,0].set_title('线图')
    axes[0,1].scatter(x, np.cos(x), alpha=0.6)
    axes[0,1].set_title('散点图')
    axes[1,0].bar(['A','B','C'], [3,7,5])
    axes[1,0].set_title('条形图')
    axes[1,1].hist(np.random.randn(500), bins=20)
    axes[1,1].set_title('直方图')

    plt.tight_layout()
    plt.savefig('subplots_demo.png', dpi=100)
    plt.show()

if __name__=="__main__":
    
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']      # 显示中文
    plt.rcParams['axes.unicode_minus'] = False        # 显示负号
    # first_pic()
    # plot_pic()
    # scatter_pic()
    # bar_pic()
    # pie_pic()
    # hist_pic()
    # boxplot_pic()
    # sub_plot_pic()
    # subplots_pic()
    # plot_save_pic()
    # plt_read_png()
    # plt_oo_pic()
    # plt_3d_pic()
    # student_socre_anlysis()
    # homework_1()
    # homework_2()
    # homework_3()
    homework_4()