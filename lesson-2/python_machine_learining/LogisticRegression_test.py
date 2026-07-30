import numpy as np
import matplotlib.pyplot as plt
def sigmoid_learn():

    z = np.linspace(-8, 8, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    plt.plot(z, sigmoid)
    plt.axhline(0.5, color='r', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    plt.title('Sigmoid 函数')
    plt.grid(True)
    plt.show()

def likelihood_function():
    x = np.linspace(0.001, 0.999, 400)
    y1 = -np.log(x)       # y=1 时的损失：预测越接近 1，损失越小
    y2 = -np.log(1 - x)   # y=0 时的损失：预测越接近 0，损失越小

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, y1, 'r'); axes[0].set_title('-log(p)  真实类别=1')
    axes[1].plot(x, y2, 'b'); axes[1].set_title('-log(1-p)  真实类别=0')
    plt.show()
if __name__=="__main__":
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']      # 显示中文
    plt.rcParams['axes.unicode_minus'] = False        # 显示负号
    # sigmoid_learn()
    likelihood_function()