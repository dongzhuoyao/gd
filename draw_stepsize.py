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

sgd_fixed = np.load("result/stepsize/SGD-fixed001.npy")
sgd_ed = np.load("result/stepsize/SGD-exponentialdecay.npy")
sgd_bt = np.load("result/stepsize/SGD-backtracking.npy")

momentum_fixed = np.load("result/stepsize/SGD-momentum09-fixed001.npy")
momentum_ed = np.load("result/stepsize/SGD-momentum09-exponentialdecay.npy")
momentum_bt = np.load("result/stepsize/SGD-momentum09-backtracking.npy")

sag_fixed = np.load("result/stepsize/SAG-fixed001.npy")
sag_ed = np.load("result/stepsize/SAG-exponentialdecay.npy")
sag_bt = np.load("result/stepsize/SAG-backtracking.npy")

svrg_fixed = np.load("result/stepsize/SAGA-fixed001.npy")
svrg_ed = np.load("result/stepsize/SAGA-exponentialdecay.npy")
svrg_bt = np.load("result/stepsize/SAGA-backtracking.npy")



optimum = 0.3500
#colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# plot with various axes scales
fig = plt.figure(1,figsize=(20,18 ))


#fig.suptitle('Step-Size Strategy Experiment', fontsize=14, fontweight='bold')

ax = fig.add_subplot(421)
sgd_fixed_handler, = ax.plot(list(sgd_fixed[0]), list(sgd_fixed[1]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
sgd_ed_handler, = ax.plot(list(sgd_ed[0]), list(sgd_ed[1]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
sgd_bt_handler, = ax.plot(list(sgd_bt[0]), list(sgd_bt[1]), color="g", linestyle="-", marker='o', markersize="4", label='Backtracking')


ax.set_title("(a).SGD training loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.legend(loc="upper right",handles=[sgd_fixed_handler,sgd_ed_handler,sgd_bt_handler])


ax = fig.add_subplot(422)
sgd_fixed_handler, = ax.plot(list(sgd_fixed[4]), list(sgd_fixed[5]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
sgd_ed_handler, = ax.plot(list(sgd_ed[4]), list(sgd_ed[5]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
sgd_bt_handler, = ax.plot(list(sgd_bt[4]), list(sgd_bt[5]), color="g", linestyle="-", marker='o',  markersize="4",label='Backtracking')


ax.set_title("(b).SGD test accuracy")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.6,0.92])
ax.legend(loc="lower right",handles=[sgd_fixed_handler,sgd_ed_handler,sgd_bt_handler])




###########################

ax = fig.add_subplot(423)
momentum_fixed_handler, = ax.plot(list(momentum_fixed[0]), list(momentum_fixed[1]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
momentum_ed_handler, = ax.plot(list(momentum_ed[0]), list(momentum_ed[1]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
momentum_bt_handler, = ax.plot(list(momentum_bt[0]), list(momentum_bt[1]), color="g", linestyle="-", marker='o', markersize="4", label='Backtracking')


ax.set_title("(c).Momentum training loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.legend(loc="upper right",handles=[momentum_fixed_handler,momentum_ed_handler,momentum_bt_handler])





ax = fig.add_subplot(424)
momentum_fixed_handler, = ax.plot(list(momentum_fixed[4]), list(momentum_fixed[5]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
momentum_ed_handler, = ax.plot(list(momentum_ed[4]), list(momentum_ed[5]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
momentum_bt_handler, = ax.plot(list(momentum_bt[4]), list(momentum_bt[5]), color="g", linestyle="-", marker='o',  markersize="4",label='Backtracking')


ax.set_title("(d).Momemtum test accuracy")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.7,0.92])
ax.legend(loc="lower right",handles=[momentum_fixed_handler,momentum_ed_handler,momentum_bt_handler])



############################

ax = fig.add_subplot(425)
sag_fixed_handler, = ax.plot(list(sag_fixed[0]), list(sag_fixed[1]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
sag_ed_handler, = ax.plot(list(sag_ed[0]), list(sag_ed[1]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
sag_bt_handler, = ax.plot(list(sag_bt[0]), list(sag_bt[1]), color="g", linestyle="-", marker='o', markersize="4", label='Backtracking')


ax.set_title("(e).SAG training loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.legend(loc="upper right",handles=[sag_fixed_handler,sag_ed_handler,sag_bt_handler])





ax = fig.add_subplot(426)
sag_fixed_handler, = ax.plot(list(sag_fixed[4]), list(sag_fixed[5]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
sag_ed_handler, = ax.plot(list(sag_ed[4]), list(sag_ed[5]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
sag_bt_handler, = ax.plot(list(sag_bt[4]), list(sag_bt[5]), color="g", linestyle="-", marker='o',  markersize="4",label='Backtracking')


ax.set_title("(f).SAG test accuracy")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.5,0.92])
ax.legend(loc="lower right",handles=[sag_fixed_handler,sag_ed_handler,sag_bt_handler])

############################



ax = fig.add_subplot(427)
svrg_fixed_handler, = ax.plot(list(svrg_fixed[0]), list(svrg_fixed[1]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
svrg_ed_handler, = ax.plot(list(svrg_ed[0]), list(svrg_ed[1]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
svrg_bt_handler, = ax.plot(list(svrg_bt[0]), list(svrg_bt[1]), color="g", linestyle="-", marker='o', markersize="4", label='Backtracking')


ax.set_title("(g).SVRG training loss")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Training Loss')
ax.legend(loc="upper right",handles=[svrg_fixed_handler,svrg_ed_handler,svrg_bt_handler])





ax = fig.add_subplot(428)
svrg_fixed_handler, = ax.plot(list(svrg_fixed[4]), list(svrg_fixed[5]), color="b", linestyle="-", marker='o', markersize="4", label='Fixed')
svrg_ed_handler, = ax.plot(list(svrg_ed[4]), list(svrg_ed[5]), color="r", linestyle="-", marker='o', markersize="4", label='Exponential Decay')
svrg_bt_handler, = ax.plot(list(svrg_bt[4]), list(svrg_bt[5]), color="g", linestyle="-", marker='o',  markersize="4",label='Backtracking')


ax.set_title("(h).SVRG test accuracy")
ax.set_xlabel('Effective Passes')
ax.set_ylabel('Test Accuracy')
ax.set_ylim([0.6,0.92])
ax.legend(loc="lower right",handles=[svrg_fixed_handler,svrg_ed_handler,svrg_bt_handler])


fig.tight_layout()
plt.savefig('stepsize.pdf', format='pdf')
plt.show()
