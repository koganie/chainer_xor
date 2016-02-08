#!/usr/bin/env python
# -*- coding: utf-8 -*-

#tutorialに書いてある内容
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#csvファイルの読み込み用
import csv

#ニューラルネットワークの定義
class NN(Chain):
    #構造
    def __init__(self):
        super(NN, self).__init__(
            l1=L.Linear(2, 10),
            l2=L.Linear(10, 10),
            l3=L.Linear(10, 1),
        )

    #順伝播
    #ReLUは正規化線形関数
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

#回帰
class Regression(Chain):
    def __init__(self, predictor):
        super(Regression, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        #誤差は平均二乗誤差
        self.loss = F.mean_squared_error(y, t)
        #平均二乗誤差とNNの出力値を返す
        return self.loss, y

#データの読み込み
def setData(fileName, dataSize):
    f = open(fileName, 'rb')
    reader = csv.reader(f)
    header = next(reader)#ヘッダは読まないで次に進む
    x_temp = []
    y_temp = []
    for v in reader:
        x_temp.append([int(v[i]) for i in range(len(v)-1)])
        y_temp.append([int(v[2])])
        if( len(x_temp) >= dataSize ):#指定のデータサイズを超えたら終わり
            break
    
    x = np.array(x_temp, np.float32).reshape(len(x_temp),len(x_temp[0]))
    y = np.array(y_temp, np.float32).reshape(len(y_temp),len(y_temp[0]))
    return x, y

if __name__=="__main__":
    
    #学習用データ, #学習用データの望ましい出力
    x_train, y_train = setData('data.txt', 4)
    
    #テスト用データ, #テスト用データの望ましい出力
    x_test, y_test = setData('data.txt', 4)
    
    #学習を行うモデルの設定
    model = Regression( NN() )
    
    #重みの調整アルゴリズムの設定
    #SGDは確率的勾配降下法
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    
    #バッチサイズはいくついっぺんに学習を行うか
    #多ければ計算時間の短縮になる
    trainBatchSize = 1
    
    #学習データの総数
    trainDataSize = len(x_train)
    #テストデータの総数
    testDataSize = len(x_test)
    
    #テストデータの目次
    testIndexes = np.arange(testDataSize)
    
    #何回か学習を行ってみる
    for epoch in range(10):
        print('epoch %d' % epoch)
        
        #学習の度に学習データの順番をばらばらにするとなんとなくいらしい
        trainIndexes = np.random.permutation(trainDataSize)
        
        #学習ブロック
        sum_train_loss = 0
        for i in range(0, trainDataSize, trainBatchSize):
            #chainer用のデータ形式に変換
            x = chainer.Variable(x_train[trainIndexes[i : i + trainBatchSize]])
            t = chainer.Variable(y_train[trainIndexes[i : i + trainBatchSize]])
            #勾配は蓄積されてしまうので伝播する前に必ず初期化する
            model.zerograds()
            #順伝播する
            loss, y = model(x, t)
            #lossは"平均"二乗誤差のため、バッチサイズ分をかける
            sum_train_loss += loss.data * trainBatchSize
            #勾配を計算する
            loss.backward()
            #重みの更新
            optimizer.update()
            
        #テストブロック
        #ここでは出力したいがために1つずつ進めているが
        #別に二乗誤差だけ求めたければいっぺんにやればよい
        testBatchSize = 1
        sum_test_loss = 0
        for i in range(0, testDataSize, testBatchSize):
            #chainer用のデータ形式に変換
            x_temp = x_test[testIndexes[i : i + testBatchSize]]
            y_temp = y_test[testIndexes[i : i + testBatchSize]]
            x = chainer.Variable(x_temp)
            t = chainer.Variable(y_temp)
            test_loss, y = model(x, t)
            #lossは"平均"二乗誤差のため、バッチサイズ分をかける
            sum_test_loss += test_loss.data * testBatchSize
            
            print("input:{0},{1} tgt:{2} => output:{3}").format(x_temp[0][0], x_temp[0][1], y_temp[0][0], y.data)
            
        #平均二乗誤差を出力してみる
        with open("output.csv", "a") as f:
            f.write("%d,%f,%f\n" % (epoch, float(sum_train_loss)/trainDataSize, float(sum_test_loss)/testDataSize) )

    #試しに1層目だけ重みの出力
    l1_W = model.predictor.l1.W.data
    l1_b = model.predictor.l1.b.data
    with open("l1_W.csv", "a") as f:
        for i in range(0, len(l1_W)):
            for j in range(0, len(l1_W[i])):
                f.write("%f," % l1_W[i][j])
            f.write("\n")
            
    with open("l1_b.csv", "a") as f:
        for i in range(0, len(l1_b)):
            f.write("%f\n" % l1_b[i])
    