# -*- coding: utf-8 -*-

# 演算用にNumPyを、プロット用にmatplotlibをimport
import numpy as np
import matplotlib.pyplot as plt

# 識別関数の本体：y=w'xを計算してるだけ
def predict(wvec,xvec):
    out=np.dot(wvec,xvec)
    if out>=0:
        res=1
    else:
        res=-1
    return [res,out]

# 学習部：識別関数に学習データを順繰りに入れて、
# 重みベクトルを更新する
def train(wvec,xvec,label):
    [res,out]=predict(wvec,xvec)
    c=0.5 # 学習係数。大き過ぎてもまともに学習してくれないので1未満ぐらいで
    if out*label<0:
        wtmp=wvec+c*label*xvec
        return wtmp
    else:
        return wvec

# 以下本体
if __name__=='__main__':
    
    item_num=100 # 学習データは100個
    loop=1000    # 収束判定は一切してないけどとりあえず1000回ループ
    init_wvec=[1,-1,1] # 重みベクトルの初期値、適当
    
    # 学習データはxy平面の第1象限と第3象限に50個ずつ
    # np.random.randomでばらつかせてみた
    x1_1=np.ones(int(item_num/2))+10*np.random.random(int(item_num/2))
    x1_2=np.ones(int(item_num/2))+10*np.random.random(int(item_num/2))
    x2_1=-np.ones(int(item_num/2))-10*np.random.random(int(item_num/2))
    x2_2=-np.ones(int(item_num/2))-10*np.random.random(int(item_num/2))
    z=np.ones(int(item_num/2)) # こいつはバイアス項、今回は無視して1に固定

    # 学習データを1つのマトリクスにまとめる
    x1=np.c_[x1_1,x1_2,z]
    x2=np.c_[x2_1,x2_2,z]
    class_x=np.array(np.r_[x1,x2])

    # 教師ラベルを1 or -1で振って1つのベクトルにまとめる
    label1=np.ones(int(item_num/2))
    label2=-1*np.ones(int(item_num/2))
    label_x=np.array(np.r_[label1,label2])

    # NumPy.ndarrayはappend()メソッドが使えないので
    # 糞コードらしく初期arrayを作ってそこに垂直に足していく策を取る
    wvec=np.vstack((init_wvec,init_wvec))
    
    # ループ回数の分だけ繰り返しつつ、重みベクトルを学習させる
    for j in range(loop):
        for i in range(item_num):
            wvec_new=train(wvec[-1],class_x[i,:],label_x[i])
            wvec=np.vstack((wvec,wvec_new))
    w=wvec[-1] # 重みベクトルを垂直に足していった最後のものを採用する
    print(w)

    # 分離直線を引く
    x_fig=range(-15,16)
    y_fig=[-(w[1]/w[0])*xi-(w[2]/w[1]) for xi in x_fig]

    # 漫然とプロットする
    plt.scatter(x1[:,0],x1[:,1],marker='o',color='g',s=100)
    plt.scatter(x2[:,0],x2[:,1],marker='s',color='b',s=100)
    plt.plot(x_fig,y_fig)
    plt.show()
