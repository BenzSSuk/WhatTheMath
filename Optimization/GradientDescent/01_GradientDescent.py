import math
import numpy as np
import matplotlib.pyplot as plt
import time

# set target
target = 10

# define function
def y(x):

    return 3*(x*x*x*x) - 5*(x*x*x) - 2*(x*x) 

# Visualize graph
# x = np.arange(-1,1,0.1)
# plt.plot(x,y(x))

## gradient descent iteration

### x_new = x_old + lr*(-dE/dx)
### x_new = x_old - lr*(dE/dx)
### E = target - y

# define E
def E(target,x):
    # E = target - y
    # E = target - ( 3x^2 + 5 )
    # E = -3x^2 -5 + target
    # abs error
    return (target - y(x))*(target - y(x)) 

def dEdx(x):
    # E =  ( -( 3*(x*x*x*x) - 5*(x*x*x) - 2*(x*x)  ) + target )^2
    # E =  ( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )^2
    # dE = 2*( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )( -12*x*x*x + 15*x*x + 4*x )
    dE = 2*( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )*( -12*x*x*x + 15*x*x + 4*x )

    return  dE

# visualize error(Not part in calculation)
x_e = np.arange(-2,3,0.01)
# plt.plot(x_e,E(target,x_e)) 
# plt.xlim([-2,0])
# plt.ylim([-100,400])
# plt.xlabel('x')
# plt.ylabel('Error')

# 
x_initial = -0.25
lr = 0.00012
lr0 = lr
nIte = 200
lr_all = []
E_all = []
x_new_all= []
# plt.figure(1)
fig, axs = plt.subplots(2)
fig.suptitle('Variable learning rate gradient descent')
axs[1].plot(x_initial,E(target,x_initial),'ro--', linewidth=2, markersize=5)
plt.pause(7)
for i in range(nIte):
    # print(f"i:{i}/{nIte}")

    if i == 0:
        x_old = x_initial

    x_new = x_old - lr*(dEdx(x_old))
    
    e_new = E(target,x_new)
    if dEdx(x_old) < 0.01 and E(target,x_old) > 10 or (np.abs(x_new - x_old) < 0.001):
        lr=lr*1.05
    elif e_new > E(target,x_old):
        lr = lr * 0.9
    elif e_new < E(target,x_old):
        lr = lr0

    # 
    print(f'x_o:{x_old}, x_new:{x_new}, lr:{lr}, dE:{dEdx(x_old)}, E:{e_new}')
    
    # axs[0].plot(x, y)
    # axs[1].plot(x, -y)

    # plt.subplot(211)
    lr_all.append(lr)
    axs[0].plot(lr_all,'r-',linewidth=1)
    axs[0].legend(['Learning rate'])
    # axs[0].hlines(lr, -2, 0, colors='k', linestyles='dotted')
    plt.ylim([0,0.002])
    plt.ylabel('Learning rate')
    plt.xlabel('iteration')

    E_all.append(e_new)
    # axs[1].plot(E_all)

    # plt.subplot(212)
    x_new_all.append(x_new)
    axs[1].plot(x_e,E(target,x_e),'-k')
    axs[1].plot(x_new_all,E_all,'ro--', linewidth=1, markersize=3)
    axs[1].vlines(x_new, -100, 400, colors='k', linestyles='dotted')
    axs[1].text(x_new,e_new-50,'x=' + str(np.around(x_new,5)) )
    axs[1].text(x_new,e_new-80,'e=' + str(np.around(e_new,5)) )
    # if i > 0:
        # plt.arrow(x_old, E(target,x_old), x_new-x_old, E(target,x_new)-E(target,x_old), head_width=0.1, head_length=0.1,color='red')
    plt.xlim([-2,0])
    plt.ylim([-100,400])
    plt.ylabel('Square error')
    plt.pause(0.1)
    # plt.show()
    
    if e_new < 0.00001:
        plt.show()
        break
    else:
        plt.cla()
    x_old = x_new
    