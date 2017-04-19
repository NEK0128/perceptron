# -*- coding: utf-8 -*-

# 演算用にNumPyを、プロット用にmatplotlibをimport
import numpy as np
import matplotlib.pyplot as plt


TRAIN_NUM = 100  # 学習データは100個
LOOP_NUM = 1000  # 収束判定は一切してないけどとりあえず1000回ループ


# 識別関数の本体：y=w'xを計算してるだけ
def predict(x_vec, w_vec):
    out = np.dot(x_vec, w_vec)
    if out >= 0:
        res = 1
    else:
        res = -1
    return res


# 学習部：識別関数に学習データを順繰りに入れて、
# 重みベクトルを更新する
def train(train_data, w_vec):
    x_vec = train_data[0:2]  # TODO:汎用化
    y = train_data[2]
    res = predict(x_vec, w_vec)
    c = 0.5  # 学習係数。大き過ぎてもまともに学習してくれないので1未満ぐらいで

    if int(res) == int(y):
        return w_vec
    else:
        return w_vec + c * y * x_vec


if __name__=='__main__':
    
    init_w_vec = [1, -1]  # 重みベクトルの初期値、適当
    
    # 学習データはxy平面の第1象限(first)と第3象限(third)に50個ずつ
    first_1 = np.ones(int(TRAIN_NUM/2)) + 10 * np.random.random(int(TRAIN_NUM/2))
    first_2 = np.ones(int(TRAIN_NUM/2)) + 10 * np.random.random(int(TRAIN_NUM/2))
    third_1 = -np.ones(int(TRAIN_NUM/2)) - 10 * np.random.random(int(TRAIN_NUM/2))
    third_2 = -np.ones(int(TRAIN_NUM/2)) - 10 * np.random.random(int(TRAIN_NUM/2))

    # 学習データを1つのマトリクスにまとめる
    first = np.c_[first_1, first_2]
    third = np.c_[third_1, third_2]

    # 教師ラベルを1 or -1で振って1つのベクトルにまとめる
    first_label = np.ones(int(TRAIN_NUM/2))
    third_label = -1 * np.ones(int(TRAIN_NUM/2))

    first_data = np.c_[first, first_label]
    third_data = np.c_[third, third_label]

    train_data = np.r_[first_data, third_data]

    w_vec = init_w_vec

    # ループ回数の分だけ繰り返しつつ、重みベクトルを学習させる
    for loop_cnt in range(LOOP_NUM):
        for train_cnt in range(TRAIN_NUM):
            w_vec = train(train_data[train_cnt, :], w_vec)

    # 分離直線を引く
    x_fig = range(-15,16)
    y_fig = [-(w_vec[1]/w_vec[0])*xi for xi in x_fig]

    # 漫然とプロットする
    plt.scatter(first[:, 0], first[:, 1],marker='o', color='g', s=100)
    plt.scatter(third[:, 0], third[:, 1],marker='s', color='b', s=100)
    plt.plot(x_fig, y_fig)
    plt.show()
