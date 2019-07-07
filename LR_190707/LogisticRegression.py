import numpy as np
import matplotlib.pyplot as plt

'''加载数据'''
def load_data(filename):
    X, y = [], []
    
    with open(filename) as f:
        for line in f:
            l = line.strip().split()
            l_float = list(map(float, l))
            
            X.append(l_float[:-1])
            y.append(l_float[-1])
            
    return np.array(X), np.array(y)

'''sigmoid'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


'''J(Θ)'''
def loss(theta, X, y, _lambda=0):
    m, n = X.shape # m个样本， n个特征
    h = sigmoid(np.dot(X, theta))
    J = -1.0 / m * (np.dot(np.log(h).T, y) + np.dot(np.log(1-h).T, 1-y))
    
    J += _lambda / (2 * m) * np.sum(np.square(theta)) # 正则项
    
    return J.flatten()[0]

'''optimize'''
def optimize(X, y, params):
    '''
    首先思考优化过程需要哪些参数：
    lr ... 学习率
    lambda ... 正则系数
    n_epoch ... 训练轮数
    thresh ... 收敛阈值
    '''
    m, n = X.shape
    
    theta = np.zeros((n, 1)) # n个特征对应n个参数
    cost = loss(theta, X, y)
    
    lr = params.get('lr', 0.01)
    _lambda = params.get('_lambda', 0)
    n_epoch = params.get('n_epoch', 1000)
    thresh = params.get('threshold', 1e-5)
    
    def _sgd(theta, cost):
        converged = False
        cnt = 0
        while cnt < n_epoch:
            if converged:
                break
            for i in range(m):
                h = sigmoid(np.dot(X[i].reshape([1, n]), theta))
                theta = theta - lr * (1.0 / m) * X[i].reshape([n, 1]) * (h - y[i]) - lr / m * _lambda * np.r_[[[0]],theta[1:]]
                cost_new = loss(theta, X, y, _lambda)
                if abs(cost_new - cost) < thresh:
                    converged = True
                    break
                cost = cost_new
            cnt += 1
        return theta, cost_new, cnt
    return _sgd(theta, cost)

def visualize(X, y, theta):
    m, n = X.shape
    for i in range(m):
        x = X[i]
        if y[i] == 1:
            plt.scatter(x[1], x[2], marker='*', color='blue', s=50)
        else:
            plt.scatter(x[1], x[2], marker='o', color='red', s=50)
    hSpots = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    theta0, theta1, theta2 = theta
    vSpots = -(theta0 + theta1 * hSpots) / theta2
    plt.plot(hSpots, vSpots, color='green', linewidth=0.5)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    
    

if __name__ == '__main__':
    
    X, y = load_data('./data/linear.txt')
    m, n = X.shape
    X = np.concatenate([np.ones([m, 1]), X], 1) # 加偏执项
    params = {
            'lr': 0.5,
            'n_epoch': 10000,
            'threshold': 1e-8,
            }
    theta, cost, cnt_ground = optimize(X, y, params)
    print(theta, cost, cnt_ground)
    visualize(X, y, theta)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    