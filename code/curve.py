#!/usr/bin/env python2.7
import numpy as np
import math
import matplotlib.pyplot as plt



def cross_validation(data, train_size):
    random.shuffle(data)
    split = zip(*[iter(data)] * 200)
    train = []
    test = []
    remain = []
    for i in range(10):
        dup = split
        test.append(dup[i])
        for j in range(10):
            if (j != i):
                for m in range(len(dup[1])):
                    remain.append(dup[j][m])
        random.shuffle(remain)
        train.append(remain[:train_size])
    return train, test



def learning_curve(DT, BT, RF, Boost, SVM, m, size):
    mean_DT = DT.mean(axis=1)
    mean_BT = BT.mean(axis=1)
    mean_RF = RF.mean(axis=1)
    mean_Boost = Boost.mean(axis=1)
    mean_SVM = SVM.mean(axis=1)
    std_DT = DT.std(axis=1)
    std_BT = BT.std(axis=1)
    std_RF = RF.std(axis=1)
    std_Boost = Boost.std(axis=1)
    std_SVM = SVM.std(axis=1)
    for i in range(len(size)):
        std_DT[i] = std_DT[i] / np.sqrt(size[i])
        std_BT[i] = std_BT[i] / np.sqrt(size[i])
        std_RF[i] = std_RF[i] / np.sqrt(size[i])
        std_Boost[i] = std_Boost[i] / np.sqrt(size[i])
        std_SVM[i] = std_SVM[i] / np.sqrt(size[i])
    plt.xlabel('Train percentage')
    plt.ylabel('Zero-one loss')
    plt.title('Zero-one loss vs. training percentage on different models')
    plt.errorbar(m, mean_DT, color='k', yerr=std_DT, label='DT', marker='.')
    plt.errorbar(m, mean_BT, color='r', yerr=std_BT, label='BT', marker='.')
    plt.errorbar(m, mean_RF, color='b', yerr=std_RF, label='RF', marker='.')
    plt.errorbar(m, mean_Boost, color='orange', yerr=std_Boost, label='Boost', marker='.')
    plt.errorbar(m, mean_SVM, color='green', yerr=std_SVM, label='SVM', marker='.')
    plt.legend(loc='center right')
    plt.show()


def main():
    if len(sys.argv) == 4:
        train_set = sys.argv[1]
        test_set = sys.argv[2]
        p = sys.argv[3]
        train = load_csv(train_set)
        test = load_csv(test_set)
        if int(p) == 1:
            loss = test_DT(train, test, 10, 10)
            print 'ZERO-ONE-LOSS-DT ', str(loss)
        if int(p) == 2:
            loss = test_BT(train, test, 50, 10, 10)
            print 'ZERO-ONE-LOSS-BT', str(loss)
        if int(p) == 3:
            loss = test_RF(train, test, 50, 10, 10)
            print 'ZERO-ONE-LOSS-RF', str(loss)
        if int(p) == 4:
            loss = test_boost(train, test, 50, 10, 10)
            print 'ZERO-ONE-LOSS-Boost', str(loss)

    if len(sys.argv) == 2:
        data_set = sys.argv[1]
        data = load_csv(data_set)
        m = [0.025, 0.05, 0.125, 0.25]
        size = [int(item * len(data)) for item in m]
        loss_DT, loss_BT, loss_RF, loss_Boost, loss_SVM = total_loss(data, size)
        print 'loss_DT', loss_DT
        print 'loss_BT', loss_BT
        print 'loss_RF', loss_RF
        print 'loss_Boost', loss_Boost
        print 'loss_SVM', loss_SVM
        print learning_curve(loss_DT, loss_BT, loss_RF, loss_Boost, loss_SVM, m, size)
        t1, p1 = t_test(loss_DT, loss_SVM)
        print 't_test_1', t1, p1
        t2, p2 = t_test(loss_Boost, loss_SVM)
        print 't_test_2', t2, p2


main()



