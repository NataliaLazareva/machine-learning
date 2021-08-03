 import numpy as np
from sklearn.datasets import load_digits
from collections import Counter
import math

digits = load_digits()
data = digits.data.astype('int32')
target = digits.target.astype('int32')

def cuttingParameters():
    N = len(data)
    ind_s = np.arange(N)
    np.random.shuffle(ind_s)
    ind_train = ind_s[0: np.int32(N * 0.8)]

    t_train = target[ind_train]
    x_train = data[ind_train]

    ind_test = ind_s[np.int32(N * 0.8): N]
    t_test = target[ind_test]
    x_test = data[ind_test]
    return x_train, t_train, x_test, t_test

def create_split_node(i, tau):
    node = {}
    node['coordInd'] = i
    node['threshold'] = tau
    node['left'] = None
    node['right'] = None
    node['isTerminal'] = False
    return node

def stop_criteria(depth, n_i_count, eps):
 d = 7
 n_i = 10
 epsilon = 0.001
 if (depth > d or n_i_count < n_i or eps < epsilon):
     return True


def get_params(data, y):

    index = 0
    I_min = 10**5
    h_si = 0
    tau = 0
    for i in range(64):
        for j in range(16):
            # left, right - номера строк
            left, right = [], []
            for k in range(len(y)):
                #print(k)
                if (data[k][i] > j):
                    left.append(y[k])
                else:
                    right.append(y[k])
            if(len(left) == 0 or len(right) == 0):
                continue
            I, h_si = calculate_I(left, right, data, y)

            if (I < I_min):
             I_min = I
             index = i
             tau = j

    return index, tau, h_si

def calculate_I(left, right, data, y):
    n_i = len(left) + len(right)
    data1 = Counter(y)

    #энтропия
    h_si = 0
    for i in data1:
        h_si += (-1) * (data1[i] / n_i) * math.log10(data1[i] / n_i)

    #слишком сложно записано
    #H(Sij) - минимизирую сумму энтропий потомков, а не максимизирую прирост информации
    left_dict = Counter(left)
    right_dict = Counter(right)
    h_sij_left = h_sij_rigth = 0
    for i in left_dict:
        h_sij_left += (-1) * (left_dict[i] / n_i) * math.log10(left_dict[i] / n_i)
    for j in right_dict:
        h_sij_rigth += (-1) * (right_dict[j] / n_i) * math.log10(right_dict[j] / n_i)

    sum = (len(left)*h_sij_left + len(right)*h_sij_rigth)/n_i

    return sum, h_si

def split_data(data, i, tau, y):

    left_data, right_data, left_target, right_target = [], [], [], []
    for j in range(len(data)):
          if (data[j][i] < tau):
              left_data.append(data[j])
              left_target.append(y[j])
          else:
              right_data.append(data[j])
              right_target.append(y[j])

    return left_data, left_target,  right_data, right_target

def create_terminal_node(y):
    y_dict = Counter(y)
    node = {}
    for i in range(10):
      node[i] = y_dict[i]/len(y) if i in y_dict  else 0

    node['isTerminal'] = True

    return node


def create_tree(data, y, d, n_i, epsilon):
    if not stop_criteria(d, n_i, epsilon):
        i, tau, epsilon = get_params(data, y)
        left_data, left_target,  right_data, right_target = split_data(data, i, tau, y)
        node = create_split_node(i, tau)
        node['left'] = create_tree(left_data, left_target, d+1, len(left_target), epsilon)
        node['right'] = create_tree(right_data, right_target, d+1, len(right_target), epsilon)

    else:
        node = create_terminal_node(y)

    return node


def get_confidence_vector(x, t, index, tau, left, right, isTerminal):


    if (isTerminal == 'False'):
        left_data, left_target, right_data, right_target = split_data(x, index, tau, t)
        get_confidence_vector(left_data, left_target, left['coordInd'], left['threshold'], left['left'], left['right'], left['isTerminal'])
        get_confidence_vector(right_data, right_target, right['coordInd'], right['threshold'], right['left'], right['right'], right['isTerminal'])

    node1 = create_terminal_node(t)
    return node1


def main():
 x_train, t_train, x_valid, t_valid = cuttingParameters()

 node_train = create_tree(x_train, t_train, 0, len(x_train), 1)
 node1, node2 = get_confidence_vector(x_valid, t_valid, node_train['coordInd'], node_train['threshold'],
                                   node_train['left'], node_train['right'], node_train['isTerminal'])

 print(node1)

if __name__ == '__main__':
    main()

