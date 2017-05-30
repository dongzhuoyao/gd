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

sgd_l1 = np.load("result/regulation_type/SGD-backtracking-regulation1e-4-l1.npy")
sgd_l2 = np.load("result/regulation_type/SGD-backtracking-regulation1e-4-l2.npy")

momentum_l1 = np.load("result/regulation_type/SGD-momentum09-backtracking-regulation1e-4-l1.npy")
momentum_l2 = np.load("result/regulation_type/SGD-momentum09-backtracking-regulation1e-4-l2.npy")



sag_l1 = np.load("result/regulation_type/SAG-backtracking-regulation1e-4-l1.npy")
sag_l2 = np.load("result/regulation_type/SAG-backtracking-regulation1e-4-l2.npy")

svrg_l1 = np.load("result/regulation_type/SVRG-lr001-regulation1e-4-l1.npy")
svrg_l2 = np.load("result/regulation_type/SVRG-lr001-regulation1e-4-l2.npy")



optimum = 0.3500
#colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# plot with various axes scales
fig = plt.figure(1,figsize=(20,10 ))

#fig.suptitle('Regulation Type Experiment', fontsize=14, fontweight='bold')

ax = fig.add_subplot(221)
sgd_l1_handler, = ax.plot(list(sgd_l1[2]), list(sgd_l1[3]), color="b", linestyle="-", marker='o', label='SGD-L1')
sgd_l2_handler, = ax.plot(list(sgd_l2[2]), list(sgd_l2[3]), color="r", linestyle="-", marker='o', label='SGD-L2')
ax.set_title("(a). SGD")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.set_ylim([0.1,0.7])
ax.legend(loc="upper right",handles=[sgd_l1_handler,sgd_l2_handler])

ax = fig.add_subplot(222)

momentum_l1_handler, = ax.plot(list(momentum_l1[2]), list(momentum_l1[3]), color="b", linestyle="-", marker='o', label='Momemtum-L1')
momentum_l2_handler, = ax.plot(list(momentum_l2[2]), list(momentum_l2[3]), color="r", linestyle="-", marker='o', label='momemtum-L2')
ax.set_title("(b). SGD with Momemtum")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
#ax.set_ylim([0,1.4])
ax.legend(loc="upper right",handles=[momentum_l1_handler,momentum_l2_handler])


ax = fig.add_subplot(223)

sag_l1_handler, = ax.plot(list(sag_l1[2]), list(sag_l1[3]), color="b", linestyle="-", marker='o', label='SAG-L1')
sag_l2_handler, = ax.plot(list(sag_l2[2]), list(sag_l2[3]), color="r", linestyle="-", marker='o', label='SAG-L2')
ax.set_title("(c). SAG")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.set_ylim([0.2,0.6])
ax.legend(loc="upper right",handles=[sag_l1_handler,sag_l2_handler])


ax = fig.add_subplot(224)

svrg_l1_handler, = ax.plot(list(svrg_l1[2]), list(svrg_l1[3]), color="b", linestyle="-", marker='o', label='SVRG-L1')
svrg_l2_handler, = ax.plot(list(svrg_l2[2]), list(svrg_l2[3]), color="r", linestyle="-", marker='o', label='SVRG-L2')
ax.set_title("(d). SVRG")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.set_ylim([0.1,0.5])
ax.legend(loc="upper right",handles=[svrg_l1_handler,svrg_l2_handler])


plt.savefig('regulation_type.pdf', format='pdf')
plt.show()
