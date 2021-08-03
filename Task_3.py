#ГРАДИЕНТНЫЙ СПУСК
#Решение регрессии с помощью градиентного спуска.

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_boston

def cutting_parameteres():
 boston = load_boston()
 fdata=boston.data
 ftarget=boston.target
 N=len(ftarget)

 data, target = normalization(fdata, ftarget)

 x_train = t_train = x_valid = t_valid = x_test = t_test = []
 ind_s = np.arange(N)
 np.random.shuffle(ind_s)
 train_size = int(N * 0.8)

 t_train = target[:train_size]
 x_train = data[:train_size]

 t_test = target[train_size:]
 x_test = data[train_size:]

 return x_train, t_train, x_test, t_test

def normalization(data, target):
 arr_data=(data-np.mean(data, axis=0))/np.std(data, axis=0)
 arr_target=(target-np.mean(target))/np.std(target)
 return arr_data, arr_target


def get_matrix(M, x_train, t_train):
 fi_set = [[0,0,0,4,3,1,0,1,0,6,2,4,0], [0,1,3,0,5,2,0,4,0,6,0,4,0], [1,0,2,0,3,0,4,0,5,0,6,0,7],
           [0,0,0,0,0,0,1,2,3,4,5,6,7], [1,2,3,4,5,6,0,0,0,0,0,0,0], [0,1,0,3,0,5,0,7,0,9,0,11,0],
           [1,0,0,8,0,0,3,0,0,6,0,0,5], [1,1,2,2,3,3,4,4,5,5,6,6,7], [0,0,0,2,0,0,0,3,0,0,0,1,0],
           [0,4,0,0,0,1,0,0,0,6,0,0,0], [5,0,0,1,0,0,0,2,0,0,0,6,0]]
 lambda_cur=0.0001

 f = np.ones((len(t_train), M))
 for i in range(0, len(t_train)):
  for j in range (1,M):
   f[i, j]= sum(np.power(x_train[i], fi_set[j], dtype=np.float))
 return f

def grad(w, f, t_train, M):
  I=np.eye(M)
  lambda_cur=0.0001
  deltaE=w.T.dot(f.T.dot(f)+lambda_cur*I)-t_train.T.dot(f)
  return deltaE

#E_cur
def get_error(w_cur, f, t_cur):
    lambda_cur = 0.0001
    y = f.dot(w_cur.T)
    E_cur=(0.5 * sum((t_cur - y) ** 2))+(lambda_cur/2)*w_cur.T@w_cur
    return E_cur

def main_part():
 iters_num = 10
 M = 11
 x_train, t_train, x_test, t_test = cutting_parameteres()
 w = np.random.randn(M)*0.1
 f = get_matrix(M, x_train, t_train)
 gamma = 0.01
 eps=10**(-5)

 E_mas = []

 for i in range(iters_num):
  g=grad(w, f, t_train, M)
  new_point = w-gamma*g
  g_norm=np.linalg.norm(g, ord=1)
  g_norm_dif=np.linalg.norm((new_point-w), ord=1)

  if g_norm < eps or g_norm_dif < eps: break
  else:w=new_point
  E_mas.append(get_error(w, f, t_train))

 f_1=get_matrix(M, x_test, t_test)
 E_test = get_error(w, f_1, t_test)

 #График
 iters_mas = np.arange(iters_num)
 plt.figure()
 plt.plot(iters_mas, E_mas, 'b.-')
 plt.show()

 return E_mas[iters_num-1], E_test

E_train, E_test = main_part()
print(E_train, E_test)





