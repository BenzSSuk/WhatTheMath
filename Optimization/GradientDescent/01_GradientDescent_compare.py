import math
import numpy as np
import matplotlib.pyplot as plt
import time

# set target
target = 10

# define function
def y(x):

    return 3*(x*x*x*x) - 5*(x*x*x) - 2*(x*x) 

## gradient descent iteration
### x_new = x_old + lr*(-dE/dx)
### x_new = x_old - lr*(dE/dx)
### E = target - y

# define E
def E(target,x):
    # Use square error
    # E = ( target - y )^2
    return (target - y(x))*(target - y(x)) 

def dEdx(x):
    # E =  ( -( 3*(x*x*x*x) - 5*(x*x*x) - 2*(x*x)  ) + target )^2
    # E =  ( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )^2
    # dE = 2*( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )( -12*x*x*x + 15*x*x + 4*x )
    dE = 2*( -3*(x*x*x*x) + 5*(x*x*x) + 2*(x*x) + target )*( -12*x*x*x + 15*x*x + 4*x )

    return  dE

# visualize error(Not part in calculation)
x_e = np.arange(-2,3,0.01)

# 
x_initial = -0.25
lr1 = 0.00012
lr0 = lr1
lr2 = lr1

nIte = 300
lr1_all = []
lr2_all = []
E_all1 = []
E_all2 = []
x_new_all1 = []
x_new_all2 = []

# plt.figure(1)
plt.close
fig, axs = plt.subplots(2)
fig.suptitle('Adaptive gradient descent')
axs[1].plot(x_initial,E(target,x_initial),'ro--', linewidth=2, markersize=5)
plt.pause(7)
for i in range(nIte):
    # print(f"i:{i}/{nIte}")

    if i == 0:
        x_old1 = x_initial
        x_old2 = x_initial

    # method 1 fix
    x_new1 = x_old1 - lr1*(dEdx(x_old1))
    e_new1 = E(target,x_new1)\

    if e_new1 > 0.00001:
        ite_1 = i
        E_all1.append(e_new1)
        lr1_all.append(lr1)
        x_new_all1.append(x_new1)


    # method 2 adaptive
    x_new2= x_old2 - lr2*(dEdx(x_old2))
    e_new2 = E(target,x_new2)
    e_old2 = E(target,x_old2)
    if dEdx(x_old2) < 0.01 and e_old2 > 10 or (np.abs(x_new2 - x_old2) < 0.001):
        lr2 = lr2*1.05
    elif e_new2 > e_old2:
        lr2 = lr2 * 0.998
    elif e_new2 < e_old2:
        lr2 = lr0

    if e_new2 > 0.00001:
        ite_2 = i
        E_all2.append(e_new2)
        lr2_all.append(lr2)
        x_new_all2.append(x_new2)
        # print(f'x_o:{x_old}, x_new:{x_new}, lr:{lr}, dE:{dEdx(x_old)}, E:{e_new}')
        
    # axs[0].plot(x, y)
    # axs[1].plot(x, -y)

    # plt.subplot(211)
    axs[0].plot(lr1_all,'b-',linewidth=1)
    axs[0].plot(lr2_all,'r-',linewidth=1)
    axs[0].legend(['Fix lr','Adaptive lr'])
    # axs[0].hlines(lr, -2, 0, colors='k', linestyles='dotted')
    plt.ylim([0,0.002])
    plt.ylabel('Learning rate')
    # plt.xlabel('iteration')

    axs[1].plot(x_e,E(target,x_e),'-k')

    # method 1
    
    axs[1].plot(x_new_all1,E_all1,'bo--', linewidth=1, markersize=5)
    # axs[1].legend(['Fix lr'])
    axs[1].vlines(x_new1, -150,300, colors='b', linestyles='dotted')
    axs[1].text(x_new1+0.05,e_new1-50,'Fix',color = 'b') 
    axs[1].text(x_new1+0.05,e_new1-80,'ite=' + str(ite_1) )
    axs[1].text(x_new1+0.05,e_new1-110,'x=' + str(np.around(x_new1,5)) )
    axs[1].text(x_new1+0.05,e_new1-140,'e=' + str(np.around(e_new1,5)) )
    

    # method 2
    axs[1].plot(x_new_all2,E_all2,'ro--', linewidth=1, markersize=3)
    # axs[1].legend(['Adaptive lr'])
    axs[1].vlines(x_new2, -150,300, colors='r', linestyles='dotted')
    axs[1].text(x_new2+0.05,e_new2+140,'Adaptive',color = 'r') 
    axs[1].text(x_new2+0.05,e_new2+110,'ite=' + str(ite_2) )
    axs[1].text(x_new2+0.05,e_new2+80,'x=' + str(np.around(x_new2,5)) )
    axs[1].text(x_new2+0.05,e_new2+50,'e=' + str(np.around(e_new2,5)) )

    # if i > 0:
        # plt.arrow(x_old, E(target,x_old), x_new-x_old, E(target,x_new)-E(target,x_old), head_width=0.1, head_length=0.1,color='red')
    plt.xlim([-2,0])
    plt.ylim([-150,300])
    plt.ylabel('Sq error')
    plt.pause(0.1)
    # plt.show()
    
    if e_new1 < 0.00001 and e_new2 < 0.00001:
        plt.pause(4)
        plt.close()
        break
    else:
        plt.cla()

    x_old1 = x_new1
    x_old2 = x_new2