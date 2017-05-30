import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
"""
# Fixing random state for reproducibility
np.random.seed(19680801)
# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))
"""

sgd = np.load("result/SGD-backtracking-l2.npy")
momentum = np.load("result/SGD-momemtum-backtracking-l2.npy")
momentum_nesterov = np.load("result/SGD-momemtum-nesterov-backtracking-l2.npy")
sag = np.load("result/SAG-backtracking-l2.npy")

saga = np.load("result/SAGA-backtracking-l2.npy")
svrg = np.load("result/SVRG-lr001-l2.npy")



#optimum = 0.3384
optimum = 0.3200
#colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# plot with various axes scales
fig = plt.figure(1,figsize=(20,15))


#fig.suptitle('Step-Size Strategy Experiment', fontsize=14, fontweight='bold')
ax = fig.add_subplot(221)
sgd_handler, = ax.plot(list(sgd[0]), list(np.log10(sgd[1]/optimum)), color="b", linestyle="-", marker='o', markersize="4", label='SGD')
momentum_handler, = ax.plot(list(momentum[0]), list(np.log10(momentum[1]/optimum)), color="r", linestyle="-", marker='o', markersize="4", label='Momentum')
momentum_nesterov_handler, = ax.plot(list(momentum_nesterov[0]), list(np.log10(momentum_nesterov[1]/optimum)), color="g", linestyle="-", marker='o', markersize="4", label='Momentum with Nesterov')
sag_handler, = ax.plot(list(sag[0]), list(np.log10(sag[1]/optimum)), color="m", linestyle="-", marker='o', markersize="4", label='SAG')

saga_handler, = ax.plot(list(saga[0]), list(np.log10(saga[1]/optimum)), color="y", linestyle="-", marker='o', markersize="4", label='SAGA')

svrg_handler, = ax.plot(list(svrg[0]), list(np.log10(svrg[1]/optimum)), color="k", linestyle="-", marker='o', markersize="4", label='SVRG')




ax.set_title("(a). Training Loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('log10(Training Loss/Optimum)')
ax.legend(loc="upper right",handles=[sgd_handler,momentum_handler,momentum_nesterov_handler,sag_handler,saga_handler,svrg_handler])


ax = fig.add_subplot(222)
sgd_handler, = ax.plot(list(sgd[2]), list(sgd[3]), color="b", linestyle="-", marker='o', markersize="4", label='SGD')
momentum_handler, = ax.plot(list(momentum[2]), list(momentum[3]), color="r", linestyle="-", marker='o', markersize="4", label='Momentum')
momentum_nesterov_handler, = ax.plot(list(momentum_nesterov[2]), list(momentum_nesterov[3]), color="g", linestyle="-", marker='o', markersize="4", label='Momentum with Nesterov')
sag_handler, = ax.plot(list(sag[2]), list(sag[3]), color="m", linestyle="-", marker='o', markersize="4", label='SAG')
saga_handler, = ax.plot(list(saga[2]), list(saga[3]), color="y", linestyle="-", marker='o', markersize="4", label='SAGA')

svrg_handler, = ax.plot(list(svrg[2]), list(svrg[3]), color="k", linestyle="-", marker='o', markersize="4", label='SVRG')


ax.set_title("(b). Validation Loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Validation Loss')
ax.legend(loc="upper right",handles=[sgd_handler,momentum_handler,momentum_nesterov_handler,sag_handler,saga_handler,svrg_handler])

ax = fig.add_subplot(223)
sgd_handler, = ax.plot(list(sgd[4]), list(sgd[5]), color="b", linestyle="-", marker='o', markersize="7", label='SGD')
momentum_handler, = ax.plot(list(momentum[4]), list(momentum[5]), color="r", linestyle="-", marker='o', markersize="4", label='Momentum')
momentum_nesterov_handler, = ax.plot(list(momentum_nesterov[4]), list(momentum_nesterov[5]), color="g", linestyle="-", marker='o', markersize="4", label='Momentum with Nesterov')
sag_handler, = ax.plot(list(sag[4]), list(sag[5]), color="m", linestyle="-", marker='o', markersize="4", label='SAG')
saga_handler, = ax.plot(list(saga[4]), list(saga[5]), color="y", linestyle="-", marker='o', markersize="4", label='SAGA')

svrg_handler, = ax.plot(list(svrg[4]), list(svrg[5]), color="k", linestyle="-", marker='o', markersize="4", label='SVRG')




ax.set_title("(c). Test Accuracy")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.8,0.92])
ax.legend(loc="lower right",handles=[sgd_handler,momentum_handler,momentum_nesterov_handler,sag_handler,saga_handler,svrg_handler])


#fig.tight_layout()
plt.savefig('method.pdf', format='pdf')
#plt.show()
