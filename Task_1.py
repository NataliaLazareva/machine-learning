#линейная регрессия
import numpy as np
import matplotlib.pyplot as plt
N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error
#1
plt.figure()
plt.plot(x,t, 'b. ')
plt.plot(x,z, 'r.-')

M=8
#матрица плана
f=np.ones((N, M))
for i in range(M):
 f[:,i]=x**i
w=np.linalg.inv(f.T@f)@f.T@t
print("w = ", w)
y_1 = f@w
plt.figure()
plt.plot(x,y_1+error, 'b. ')
plt.plot(x,z, 'r.-')

M=100
f=np.ones((N, M))
for i in range(M):
 f[:,i]=x**i
w=np.linalg.inv(f.T@f)@f.T@t
y_2 = f@w
plt.figure()
plt.plot(x,y_2+error, 'b. ')
plt.plot(x,z, 'r.-')

M_1=[i for i in range(1, 100)]
E_w=[]
for i in range(1, 100):
  f = np.ones((N, i))
  for k in range(1, i):
   f[:, k] = x**k
  w=np.linalg.inv(f.T@f)@f.T@t
  y = f@w
  E_w.append(0.5*sum((t-y)**2))

plt.figure()
plt.plot(M_1, E_w, 'r.-')
plt.show()