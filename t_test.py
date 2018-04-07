#!/usr/bin/env python2.7
import numpy as np
from scipy import stats



def t_test(x, y):
    t = np.zeros((4, 1))
    p_value = np.zeros((4, 1))
    for i in range(4):
        t[i], p_value[i] = stats.ttest_ind(x[i, :], y[i, :], equal_var=True)
    return t, p_value


t1, p1 = t_test(loss_DT, loss_SVM)
print 't_test_1', t1, p1
t2, p2 = t_test(loss_Boost, loss_SVM)
print 't_test_2', t2, p2

