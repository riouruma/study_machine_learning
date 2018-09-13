import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('click.csv', delimiter = ',', skiprows = 1)
train_x = train[:,0]
train_y = train[:,1]

# パラメータを初期化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 予測関数
def f(x):
    return theta0 + theta1 * x

# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 標準化
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

#プロット
# plt.plot(train_z, train_y, 'o')
# plt.show()

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
error = E(train_z, train_y)

while diff > 1e-2:
    # 更新結果を一時変数に保存
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    # パラメータを更新
    theta0 = tmp0
    theta1 = tmp1

    # 前回の誤差との差分を計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error

    # ログの出力
    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))

x = np.linspace(-3, 3, 100)

print(f(standardize(100)))
print(f(standardize(200)))
print(f(standardize(300)))

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()
