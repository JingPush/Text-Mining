import sys
import csv
import numpy as np
from numpy import linalg as LA
from collections import Counter
import random
import math
import matplotlib.pyplot as plt
from scipy import stats



def load_csv(input_file):
    file = open(input_file, 'rU')
    data = list(csv.reader(file, delimiter='\t'))
    return data


def word(data):
    punc1 = ("~`!@#$%^&*()_-+=[]{}\|;:',<.>/?")
    punc2 = ('"')
    wordsbag = []
    words = zip(*data)[2]
    words = [item.lower().translate(None, punc1).translate(None, punc2) for item in words]
    words = [item.split() for item in words]
    for line in words:
        wordsbag.extend(set(line))
    return [words, wordsbag]


def count_attr(data):
    words, wordsbag = word(data)
    c = Counter(wordsbag)
    length = min(1100, len(wordsbag))
    feature = c.most_common(length)[100:length]
    return feature


def summarize_feature(data, feature):
    words, wordsbag = word(data)
    feature_value = np.zeros((len(data), len(feature)))
    for i in range(len(words)):
        for j in range(len(feature)):
            if (feature[j][0] in words[i]):
                feature_value[i, j] = 1
    return feature_value


def get_gini(y):
    gini = 0
    length = float(len(y))
    if len(y) > 0:
        gini = 1 - (np.count_nonzero(y == 0)/length)**2 - (np.count_nonzero(y == 1)/length)**2
    return gini


def get_gain(upper, left, right):
    gain = get_gini(upper) - get_gini(left) * len(left)/len(upper) - get_gini(right) * len(right)/len(upper)
    return gain


def split_data(dataset, index):
    x = dataset['x']
    y = dataset['y']
    left_x = x[x[:, index] == 1]
    left_y = y[x[:, index] == 1]
    right_x = x[x[:, index] == 0]
    right_y = y[x[:, index] == 0]
    return left_x, left_y, right_x, right_y



def split_best_tree(dataset):
    max_gain = 0
    x = dataset['x']
    y = dataset['y']
    best_left, best_right, index = None, None, None
    for i in range(len(x[0, :])):
        left_x, left_y, right_x, right_y = split_data(dataset, i)
        gain = get_gain(y, left_y, right_y)
        if gain > max_gain:
            max_gain = gain
            index = i
            best_left = {'x': left_x, 'y': left_y}
            best_right = {'x': right_x, 'y': right_y}
    return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}




def to_terminal(node):
    y = node['y']
    pred_y = 0
    if np.count_nonzero(y == 1) >= np.count_nonzero(y == 0):
        pred_y = 1
    return pred_y



def split_tree(node, depth, max_depth, min_size):

    if node['pivot'] == None :
        node['left'] = to_terminal(node)
        node['right'] = to_terminal(node)
        return

    left = node['left']
    right = node['right']
    #print len(left['y']), len(right['y']), node['pivot']

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left['y']) <= min_size:
        node['left'] = to_terminal(left)
        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_tree(right)
            split_tree(node['right'], depth+1, max_depth, min_size)
        return

    else:
        node['left'] = split_best_tree(left)
        split_tree(node['left'], depth+1, max_depth, min_size)

        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_tree(right)
            split_tree(node['right'], depth+1, max_depth, min_size)


def build_tree(train, feature, max_depth, min_size):
    x = summarize_feature(train, feature)
    class_label = np.asarray([int(item) for item in zip(*train)[1]])
    train = {'x': x, 'y': class_label}
    root = split_best_tree(train)
    split_tree(root, 1, max_depth, min_size)
    return root


def pred(node, x):
    if node['pivot'] == None:
        return node['left']
    if x[node['pivot']] == 1:
        if isinstance(node['left'], dict):
            return pred(node['left'], x)
        else:
            return node['left']
    if x[node['pivot']] == 0:
        if isinstance(node['right'], dict):
            return pred(node['right'], x)
        else:
            return node['right']


def pred_tree(tree, feature, test):
    x = summarize_feature(test, feature)
    pred_y = np.zeros((len(test), 1))
    for i in range(len(test)):
        pred_y[i] = pred(tree, x[i, :])
    return pred_y


def test_DT(train, test, max_depth, min_size):
    feature = count_attr(train)
    class_label = np.asarray([int(item) for item in zip(*test)[1]])
    tree = build_tree(train, feature, max_depth, min_size)
    pred_class = pred_tree(tree, feature, test)
    loss = cal_loss(pred_class, class_label)
    return loss


def cal_loss(pred, class_label):
    loss = 0
    for i in range(len(pred)):
        if pred[i] != class_label[i]:
            loss += 1
    loss = loss / float(len(pred))
    return loss




def pred_BT(train, test, num, max_depth, min_size):
    feature = count_attr(train)
    pred_total = np.zeros((len(test), num))
    pred_final = np.zeros((len(test), 1))
    for i in range(num):
        new_train = [random.choice(train) for j in range(len(train))]
        tree = build_tree(new_train, feature, max_depth, min_size)
        pred_total[:, i] = pred_tree(tree, feature, test)[:, 0]
    for i in range(len(test)):
        if np.count_nonzero(pred_total[i, :]) >= num/2:
            pred_final[i] = 1
    return pred_final



def test_BT(train, test, num, max_depth, min_size):
    class_label = np.asarray([int(item) for item in zip(*test)[1]])
    pred_class = pred_BT(train, test, num, max_depth, min_size)
    loss = cal_loss(pred_class, class_label)
    return loss



def split_best_forest(dataset, k):
    y = dataset['y']
    x = dataset['x']
    max_gain = 0
    best_left, best_right, index = None, None, None
    select_index = random.sample(list(range(len(x[0, :]))), k)
    for i in range(k):
        m = select_index[i]
        left_x, left_y, right_x, right_y = split_data(dataset, m)
        gain = get_gain(y, left_y, right_y)
        if gain > max_gain:
            max_gain = gain
            index = m
            best_left = {'x': left_x, 'y': left_y}
            best_right = {'x': right_x, 'y': right_y}
    return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}
    


def split_RF(node, depth, max_depth, min_size, k):

    if node['pivot'] == None :
        node['left'] = to_terminal(node)
        node['right'] = to_terminal(node)
        return

    left = node['left']
    right = node['right']

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left['y']) <= min_size:
        node['left'] = to_terminal(left)
        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_forest(right, k)
            split_RF(node['right'], depth+1, max_depth, min_size, k)
        return

    else:
        node['left'] = split_best_forest(left, k)
        split_RF(node['left'], depth+1, max_depth, min_size, k)

        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_forest(right, k)
            split_RF(node['right'], depth+1, max_depth, min_size, k)



def build_RF(train, max_depth, min_size, feature, k):
    x = summarize_feature(train, feature)
    class_label = np.asarray([int(item) for item in zip(*train)[1]])
    train = {'x': x, 'y': class_label}
    root = split_best_forest(train, k)
    split_RF(root, 1, max_depth, min_size, k)
    return root



def pred_RF(train, test, num, max_depth, min_size):
    pred_total = np.zeros((len(test), num))
    pred_final = np.zeros((len(test), 1))
    feature = count_attr(train)
    k = int(math.sqrt(len(feature)))
    for i in range(num):
        new_train = [random.choice(train) for j in range(len(train))]
        tree = build_RF(new_train, max_depth, min_size, feature, k)
        pred_total[:, i] = pred_tree(tree, feature, test)[:, 0]
    for i in range(len(test)):
        if np.count_nonzero(pred_total[i, :]) >= num/2:
            pred_final[i] = 1
    return pred_final



def test_RF(train, test, num, max_depth, min_size):
    class_label = np.asarray([int(item) for item in zip(*test)[1]])
    pred_class = pred_RF(train, test, num, max_depth, min_size)
    loss = cal_loss(pred_class, class_label)
    return loss




def boost_gini(y, w):
    gini = 0
    if len(y) > 0:
        sum_total = sum(w)
        sum_w_1 = sum(w[y == 1])
        sum_w_0 = sum(w[y == 0])
        gini = 1 - (sum_w_1 / sum_total)**2 - (sum_w_0/sum_total)**2
    return gini


def boost_gain(upper_y, upper_w, left_y, left_w, right_y, right_w):
    gain_left = boost_gini(left_y, left_w) * sum(left_w) / sum(upper_w)
    gain_right = boost_gini(right_y, right_w) * sum(right_w) / sum(upper_w)
    gain = boost_gini(upper_y, upper_w) - gain_left - gain_right
    return gain



def split_boost_data(dataset, index):
    x = dataset['x']
    y = dataset['y']
    w = dataset['w']
    left_x = x[x[:, index] == 1]
    left_y = y[x[:, index] == 1]
    left_w = w[x[:, index] == 1]
    right_x = x[x[:, index] == 0]
    right_y = y[x[:, index] == 0]
    right_w = w[x[:, index] == 0]
    return left_x, left_y, left_w, right_x, right_y, right_w





def split_best_boost(dataset):
    max_gain = 0
    x = dataset['x']
    y = dataset['y']
    w = dataset['w']
    best_left, best_right, index = None, None, None
    for i in range(len(x[0, :])):
        left_x, left_y, left_w, right_x, right_y, right_w = split_boost_data(dataset, i)
        gain = boost_gain(y, w, left_y, left_w, right_y, right_w)
        if gain > max_gain:
            max_gain = gain
            index = i
            best_left = {'x': left_x, 'y': left_y, 'w': left_w}
            best_right = {'x': right_x, 'y': right_y, 'w': right_w}
    return {'left': best_left, 'right': best_right, 'pivot': index, 'y': y}



def split_boost_tree(node, depth, max_depth, min_size):

    if node['pivot'] == None :
        node['left'] = to_terminal(node)
        node['right'] = to_terminal(node)
        return

    left = node['left']
    right = node['right']
    #print len(left['y']), len(right['y']), node['pivot']

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left['y']) <= min_size:
        node['left'] = to_terminal(left)
        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_boost(right)
            split_boost_tree(node['right'], depth+1, max_depth, min_size)
        return

    else:
        node['left'] = split_best_boost(left)
        split_boost_tree(node['left'], depth+1, max_depth, min_size)

        if len(right['y']) <= min_size:
            node['right'] = to_terminal(right)
            return
        else:
            node['right'] = split_best_boost(right)
            split_boost_tree(node['right'], depth+1, max_depth, min_size)




def build_boost_tree(train, w, feature, max_depth, min_size):
    x = summarize_feature(train, feature)
    class_label = np.asarray([int(item) for item in zip(*train)[1]])
    train = {'x': x, 'y': class_label, 'w': w}
    root = split_best_boost(train)
    split_boost_tree(root, 1, max_depth, min_size)
    return root





def train_boost(train, num, max_depth, min_size):
    feature = count_attr(train)
    true_y = np.asarray([int(item) for item in zip(*train)[1]])
    true_y[true_y == 0] = -1
    w = np.full((len(train), num+1), 1/float(len(train)))
    alpha = np.zeros((num, 1))
    total_tree = []
    for t in range(num):
        w_temp = w[:, t]
        tree = build_boost_tree(train, w_temp, feature, max_depth, min_size)
        total_tree.append(tree)
        pred_y = pred_tree(tree, feature, train)
        pred_y[pred_y == 0] = -1
        sum_error = sum(w_temp[(true_y != pred_y.T)[0]])
        if sum_error == 0:
            alpha[t] = 100
            break
        alpha[t] = 0.5 * math.log((1 - sum_error) / sum_error)
        if alpha[t] == 0:
            break
        for i in range(len(train)):
            w[i, t+1] = w_temp[i] * math.exp(- alpha[t] * true_y[i] * pred_y[i])
        w[:, t+1] = w[:, t+1]/sum(w[:, t+1])
    return alpha, total_tree




def test_boost(train, test, num, max_depth, min_size):
    alpha, total_tree = train_boost(train, num, max_depth, min_size)
    pred_total = np.zeros((len(test), num))
    feature = count_attr(train)
    true_y = np.asarray([int(item) for item in zip(*test)[1]])
    true_y[true_y == 0] = -1
    for i in range(len(total_tree)):
        tree = total_tree[i]
        pred_total[:, i] = pred_tree(tree, feature, test)[:, 0]
    pred_total[pred_total == 0] = -1
    loss = 0
    for i in range(len(test)):
        p = np.dot(alpha[:, 0], pred_total[i, :])
        if p*true_y[i] < 0:
            loss += 1
    loss = loss / float(len(test))
    return loss



def svms_feature(data, feature, n):
    feature_value = summarize_feature(data, feature)
    svms_feature = np.zeros((len(data), len(feature)+1))
    svms_feature[:, 0] = 1
    svms_feature[:, 1:len(feature)+1] = feature_value
    return svms_feature



def learn_SVMS(data, n):
    feature = count_attr(data)
    x = svms_feature(data, feature, n)
    class_label = [int(item) for item in zip(*data)[1]]
    for i in range(len(class_label)):
        if(class_label[i] == 0):
            class_label[i] = -1
    w = np.zeros((101, len(x[1])))
    w[0, 0] = 1
    count = 0
    tol = 1
    gradient = np.zeros((len(x[1]), 1))
    while ((count < 100) & (tol > 1e-6)):
        for j in range(len(gradient)):
            sub = 0
            for i in range(len(data)):
                y_hat = np.dot(w[count], x[i])
                if (y_hat * class_label[i] < 1):
                    sub += class_label[i] * x[i, j]
            gradient[j] = 0.01 * w[count, j] - sub/float(len(data))
            w[count+1, j] = w[count, j] - 0.5 * gradient[j]
        tol = abs(LA.norm(w[count + 1, :]-w[count, :]))
        count += 1
    return w[count-1, :]



def test_SVMS(train, test, n):
    feature = count_attr(train)
    x = svms_feature(test, feature, n)
    w = learn_SVMS(train, n)
    class_label = [int(item) for item in zip(*test)[1]]
    loss = 0
    pred_label = np.zeros((len(test), 1))
    for i in range(len(test)):
        p1 = np.dot(w, x[i])
        if(p1 >= 0):
            pred_label[i] = 1
        if (pred_label[i] != class_label[i]):
            loss += 1
    loss = float(loss)/float(len(test))
    return loss




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


def total_loss(data, size):
    loss_DT = np.zeros((len(size), 10))
    loss_BT = np.zeros((len(size), 10))
    loss_RF = np.zeros((len(size), 10))
    loss_Boost = np.zeros((len(size), 10))
    loss_SVM = np.zeros((len(size), 10))
    for i in range(len(size)):
        train, test = cross_validation(data, size[i])
        for j in range(10):
            print j
            loss_DT[i, j] = test_DT(train[j], test[j], 10, 10)
            print 'DT', loss_DT[i, j]
            loss_BT[i, j] = test_BT(train[j], test[j], 50, 10, 10)
            print 'BT', loss_BT[i, j]
            loss_RF[i, j] = test_RF(train[j], test[j], 50, 10, 10)
            print 'RF', loss_RF[i, j]
            loss_Boost[i, j] = test_boost(train[j], test[j], 50, 10, 10)
            print 'Boost', loss_Boost[i, j]
            loss_SVM[i, j] = test_SVMS(train[j], test[j], 2)
            print 'SVM', loss_SVM[i, j]
    return loss_DT, loss_BT, loss_RF, loss_Boost, loss_SVM



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



def t_test(x, y):
    t = np.zeros((4, 1))
    p_value = np.zeros((4, 1))
    for i in range(4):
        t[i], p_value[i] = stats.ttest_ind(x[i, :], y[i, :], equal_var=True)
    return t, p_value




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