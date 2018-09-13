import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# パラメータを初期化
theta = np.random.rand(3)

# 標準化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)

def standardize(x):
  return (x - mu) / sigma

train_z = standardize(train_x)

# x0を加える
def to_matrix(x):
  x0 = np.ones([x.shape[0], 1])
  return np.hstack([x0, x])

X = to_matrix(train_z)

# シグモイド関数
def f(x):
  return 1 / (1 + np.exp(-np.dot(x, theta)))

# 分類関数
def classify(x):
  return (f(x) >= 0.5).astype(np.int)

# 学習率
ETA = 1e-3

# 繰り返し回数
epoch = 5000

# 学習を繰り返す
for _ in range(epoch):
  theta = theta - ETA * np.dot(f(X) - train_y, X)

x0 = np.linspace(-2, 2, 100)

# plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
# plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
# plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle = 'dashed')
# plt.show()


print(classify(to_matrix(standardize([
  [200, 100], # 200x100 の横長画像
  [100, 200]  # 100x200 の縦長画像
]))))