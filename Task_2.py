#полиномиальная регрессия
import numpy as np
import matplotlib.pyplot as plt
import random

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error
M = 9

x_train=t_train=x_valid=t_valid=x_test=t_test=[]

#Перетасовка и присваивание значений
ind_s = np.arange(N)
np.random.shuffle(ind_s)
ind_train = ind_s[0: np.int32(N * 0.8)]
x_train = x[ind_train]
t_train = t[ind_train]

ind_valid = ind_s[np.int32(N * 0.8): np.int32(N * 0.9)]
x_valid = x[ind_valid]
t_valid = t[ind_valid]

ind_test = ind_s[np.int32(N * 0.9): N]
x_test = x[ind_test]
t_test = t[ind_test]

#Определение коэффициентов регуляризации
lambda_set=[0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2000]

#Определение базисных функций
fi_set=[np.sin, np.cos, np.tan, lambda x:x**2, lambda x:x**3, lambda x:x**4, lambda x:x**2+x+1, lambda x:np.sin(x)*np.cos(x),
lambda x:np.sin(x**2)*np.cos(x**2), np.exp, lambda x:x**3+x**2+x+1]

def get_current_lambda():
    return(random.choice(lambda_set))

def get_current_fi():
    return random.sample(fi_set,k=M)
#w
def get_model_param(fi_cur, lambda_cur):
    f = np.ones((len(x_train), M))
    I=np.identity(M)
    for i in range(M):
        f[:, i] = fi_cur[i](x_train)
    w = np.linalg.inv(f.T @ f+lambda_cur*I) @ f.T @ t_train
    return w

#E_cur
def get_error(fi_cur, lambda_cur, w_cur, x_1, t_cur):
    f = np.ones((len(x_1), M))
    for i in range(M):
        f[:, i] = fi_cur[i](x_1)
    y = f @ w_cur
    E_cur=(0.5 * sum((t_cur - y) ** 2))+(lambda_cur/2)*w_cur.T@w_cur
    return E_cur if len(t_cur)<N else y


iters_num=1000
E_min=10**10
for i in range(iters_num):
    lambda_cur=get_current_lambda()
    fi_cur=get_current_fi()
    w_cur=get_model_param(fi_cur, lambda_cur)
    E_cur = get_error(fi_cur, lambda_cur, w_cur, x_valid, t_valid)
    if E_cur<E_min:
        E_min=E_cur
        lambda_best=lambda_cur
        fi_best=fi_cur
        w_best=w_cur
E_model=get_error(fi_best, lambda_best, w_best, x_test, t_test)
y_model_1=get_error(fi_best, lambda_best, w_best, x, t)
print(E_model, lambda_best, fi_best, sep='\n')

plt.figure()
plt.plot(x,t, 'b. ')
plt.plot(x,z, 'r.-')
plt.plot(x,y_model_1, 'g.-')
plt.show()
